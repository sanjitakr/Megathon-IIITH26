"""Evaluation report (spec 12 / 12.1) -> docs/evaluation_report.md and data/models/evaluation.json.

Simulator-based numbers are reported as *agreement with the reference simulator*, never
as real-world accuracy. The value model (real sales, if trained on them) is the only
component evaluated against real outcomes.

    python -m ml.evaluate [--planner-homes 60] [--shift-homes 1500]
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from engine.carbon import EmissionSet, PriceSet, annual_bill
from engine.currency import money
from engine.inputs import CustomerInput, CustomiseOptions, Usage
from gen.labels import label_many, perturbed_parameters
from ml.train import load_dataset, metrics
from sim.parameters import MODELS_DIR, PROCESSED_DIR, ROOT


def shift_test(lab: pd.DataFrame, X: pd.DataFrame, props: pd.DataFrame, n: int) -> dict:
    """Train on simulator version A (already done), test on version B (perturbed parameters)."""
    bundle = joblib.load(MODELS_DIR / "baseline_models.joblib")["models"]
    te = lab.index[lab.split == "test"][:n]
    rows = props.set_index("property_id").loc[lab.property_id[te]].reset_index()
    hidden = pd.read_parquet(PROCESSED_DIR / "hidden_state.parquet").set_index("property_id")
    rows = rows.join(hidden, on="property_id").to_dict("records")
    with perturbed_parameters(0.2, seed=3):
        lab_b = pd.DataFrame(label_many(rows))
    Xt = X.loc[te]
    phys_e = np.log(Xt.phys_elec.clip(lower=50).values)
    pred_e = np.exp(phys_e + bundle["electricity"]["model"].predict(Xt))
    phys_g = np.log1p(Xt.phys_gas.values)
    pred_g = np.expm1(phys_g + bundle["gas"]["model"].predict(Xt))
    has = lab_b.clean_gas_kwh.values > 0
    return {
        "n": int(len(te)),
        "electricity_vs_version_A_clean": metrics(lab.loc[te, "clean_electricity_kwh"].values, pred_e),
        "electricity_vs_version_B_clean": metrics(lab_b.clean_electricity_kwh.values, pred_e),
        "gas_vs_version_A_clean": metrics(lab.loc[te, "clean_gas_kwh"].values[has], pred_g[has]),
        "gas_vs_version_B_clean": metrics(lab_b.clean_gas_kwh.values[has], pred_g[has]),
    }


def _input_from_row(form_json: str, lab_row) -> CustomerInput:
    form = json.loads(form_json)
    elec = json.loads(lab_row.obs_monthly_electricity)
    gas = json.loads(lab_row.obs_monthly_gas)
    form["electricity"] = Usage(monthly_kwh=elec)
    if any(g for g in gas if g):
        form["gas"] = Usage(monthly_kwh=gas)
    return CustomerInput(**form)


def planner_eval(lab: pd.DataFrame, props: pd.DataFrame, n: int, budget: float = 6000) -> dict:
    """Optimality gap of rank-and-fill greedy vs full enumeration, and stage-2 runtime."""
    from engine.pipeline import analyse, get_stage1
    from engine.validation import validate

    te = lab[lab.split == "test"].head(n)
    forms = props.set_index("property_id").loc[te.property_id, "form_json"]
    gaps, t_stage1, t_stage2, n_ok, n_abstain = [], [], [], 0, 0
    ps, es = PriceSet.from_options(), EmissionSet.from_options()
    for (_, row), form in zip(te.iterrows(), forms):
        inp = _input_from_row(form, row)
        t0 = time.perf_counter()
        res = analyse(inp, CustomiseOptions(goal="cut_bills", budget=budget), with_ranges=False)
        if res["status"] != "ok":
            n_abstain += 1
            continue
        t_stage1.append(res["timing_ms"]["stage1"])
        t1 = time.perf_counter()
        analyse(inp, CustomiseOptions(goal="cut_bills", budget=budget * 0.5), with_ranges=False)
        t_stage2.append(1000 * (time.perf_counter() - t1))
        ctx = get_stage1(inp, validate(inp))["ctx"]
        best = res["plan"]["best"]
        best_val = best["bill_saving_gbp"] if best else 0.0
        # greedy: add by standalone bill saving per GBP until the budget is used
        order = sorted(ctx.feasible, key=lambda r: -ctx.standalone[r]["saving_per_gbp"])
        chosen, spent = [], 0.0
        for rid in order:
            c = ctx.get([rid]).cost[1] if ctx.get([rid]) else None
            if c is None or spent + c > budget or ctx.standalone[rid]["saving_per_gbp"] <= 0:
                continue
            trial = ctx.get(chosen + [rid])
            if trial is None:
                continue
            chosen.append(rid)
            spent += c
        g_ev = ctx.get(chosen)
        greedy_val = annual_bill(ctx.base.purchased, ctx.base.export, ps) - annual_bill(g_ev.purchased, g_ev.export, ps)
        if best_val > 0:
            gaps.append(max(0.0, (best_val - greedy_val) / best_val))
        n_ok += 1
    return {
        "homes": n_ok, "abstained": n_abstain, "budget_gbp": budget, "goal": "cut_bills",
        "greedy_mean_optimality_gap_pct": round(100 * float(np.mean(gaps)), 2) if gaps else None,
        "greedy_share_suboptimal": round(float(np.mean(np.array(gaps) > 1e-6)), 3) if gaps else None,
        "stage1_ms_median": round(float(np.median(t_stage1)), 1) if t_stage1 else None,
        "stage2_ms_median": round(float(np.median(t_stage2)), 2) if t_stage2 else None,
    }


def score_checks() -> dict:
    """Monotonicity per component and reproducibility of the score."""
    from engine.score import compute_score
    from sim.home import HomeState

    s = HomeState(property_type="semi", age_band="D", floor_area=90, region="london", storeys=2, occupants=2.5)
    kw = dict(sap_points=55, band="D", efficiency_basis="recorded", intensity_kwh_m2=180, intensity_basis="metered", generation_kwh=0, gross_electricity_kwh=3000)
    base = compute_score(s, **kw)["score"]
    checks = {}
    for field, worse, better in (("loft", "under_100", "over_250"), ("wall_type", "cavity_unfilled", "cavity_filled"),
                                 ("glazing", "single", "double_modern"), ("controls", "none", "smart_trvs")):
        a, b = s.copy(), s.copy()
        setattr(a, field, worse)
        setattr(b, field, better)
        checks[field] = compute_score(b, **kw)["score"] >= compute_score(a, **kw)["score"]
    checks["intensity"] = compute_score(s, **{**kw, "intensity_kwh_m2": 120})["score"] >= compute_score(s, **{**kw, "intensity_kwh_m2": 300})["score"]
    checks["efficiency"] = compute_score(s, **{**kw, "sap_points": 75})["score"] >= base
    checks["reproducible"] = compute_score(s, **kw)["score"] == base
    return checks


def write_report(ev: dict) -> None:
    m = ev["training_metrics"]
    e = m["energy"]
    L = ["# Evaluation report", "",
         f"Generated {ev['generated']}. Labels come from our own simulator: unless stated otherwise every metric below is "
         "**agreement with the reference simulator**, not real-world accuracy (spec 12.1).", ""]
    L += ["## Baseline energy (agreement with the reference simulator, observed with noise and missing months)", "",
          "| fuel / split | MAE kWh | MAPE % | R2 | 80% interval coverage | mean width kWh | baseline MAPE (type x age mean) |", "|---|---|---|---|---|---|---|"]
    for k in ("electricity_val", "electricity_test", "electricity_ood_test", "gas_val", "gas_test", "gas_ood_test"):
        if k not in e:
            continue
        r = e[k]
        L.append(f"| {k} | {r['mae']:,.0f} | {r['mape']:.1f} | {r['r2']:.3f} | {r['coverage_80']:.3f} | {r['mean_width']:,.0f} | {r['baseline_type_age_mean']['mape']:.1f} |")
    L += ["", "OOD = held-out region (Wales) and age band B (1900-1929), never seen in training.", "", "### Ablations (test split)", "", "| variant | MAE | MAPE % | R2 |", "|---|---|---|---|"]
    for k, r in e["ablations_test"].items():
        L.append(f"| {k} | {r['mae']:,.0f} | {r['mape']:.1f} | {r['r2']:.3f} |")
    L.append(f"| full model (physics features + residual learning) | {e['electricity_test']['mae']:,.0f} | {e['electricity_test']['mape']:.1f} | {e['electricity_test']['r2']:.3f} |")
    L += ["", f"Optional ensemble members: {json.dumps(e['ensemble_members_test'])}", ""]
    L += ["### Fairness: electricity MAPE % by group (test + OOD)", ""]
    for g, vals in e["fairness_electricity_mape"].items():
        L.append(f"- **{g}**: " + ", ".join(f"{k} {v:.1f}" for k, v in vals.items()))
    if "shift_test" in ev:
        s = ev["shift_test"]
        L += ["", "## Shift test: trained on simulator A, tested on simulator B (parameters perturbed up to 20%)", "",
              "| target | MAE | MAPE % | R2 |", "|---|---|---|---|"]
        for k in ("electricity_vs_version_A_clean", "electricity_vs_version_B_clean", "gas_vs_version_A_clean", "gas_vs_version_B_clean"):
            r = s[k]
            L.append(f"| {k} | {r['mae']:,.0f} | {r['mape']:.1f} | {r['r2']:.3f} |")
        L += ["", "This shows sensitivity to simulator misspecification, not which version is true."]
    L += ["", "## Feasibility (precision / recall on the rule labels, test split)", "", "| retrofit | prevalence | classifier P/R | rules on defaults-filled inputs P/R | majority P/R |", "|---|---|---|---|---|"]
    for rid, r in m["feasibility"].items():
        L.append(f"| {rid} | {r['prevalence']:.2f} | {r['classifier']['precision']:.2f} / {r['classifier']['recall']:.2f} | "
                 f"{r['rules_on_filled_inputs']['precision']:.2f} / {r['rules_on_filled_inputs']['recall']:.2f} | {r['majority']['precision']:.2f} / {r['majority']['recall']:.2f} |")
    L += ["", "## Per-retrofit saving surrogates (MAE kWh vs simulator reference; baseline = fixed percentage like the brief's illustrative figures)", "",
          "| retrofit | fuel | surrogate MAE | fixed-% baseline MAE | n |", "|---|---|---|---|---|"]
    for rid, r in m["savings_surrogates"].items():
        for k, v in r.items():
            L.append(f"| {rid} | {k} | {v['surrogate_mae']:,.0f} | {v['fixed_pct_baseline_mae']:,.0f} | {v['n_test']} |")
    r = m["ranking"]
    L += ["", "## Retrofit ranking", "", f"NDCG@3 LambdaRank {r['ndcg@3_lambdarank']:.3f} vs heuristic (mean saving per unit of money) {r['ndcg@3_heuristic_mean_saving_per_gbp']:.3f} over {r['n_test_homes']} test homes. "
          "The engine itself ranks by exact simulated saving per unit cost within dependency tiers; LambdaRank is the ML comparison."]
    ip = m["improvement_potential"]
    L += ["", "## Improvement potential", "", f"Spearman (model) {ip['spearman_model']:.3f}; on homes with a known EPC band: model {ip['spearman_model_on_same_known_subset']:.3f} vs EPC-band order {ip['spearman_epc_band_order_on_known']:.3f}. MAE {ip['mae']:.1f} percentage points."]
    if "planner" in ev:
        p = ev["planner"]
        L += ["", "## Planner", "", f"Full enumeration vs rank-and-fill greedy (goal cut bills, budget {money(p['budget_gbp'])}, {p['homes']} homes): "
              f"greedy mean optimality gap {p['greedy_mean_optimality_gap_pct']}%, greedy suboptimal in {p['greedy_share_suboptimal']} of homes. "
              f"Median stage-1 {p['stage1_ms_median']} ms, stage-2 re-rank {p['stage2_ms_median']} ms."]
    L += ["", "## Score", "", "Monotonicity and reproducibility checks: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in ev["score_checks"].items())]
    if ev.get("value"):
        v = ev["value"]
        L += ["", "## Property value model", "", f"Basis: **{v.get('basis')}**. Test MAPE {v.get('mape_pct')}%, P10-P90 coverage {v.get('coverage_80')}, baseline (mean price by region and type) MAPE {v.get('baseline_mape_pct')}%.",
              "The value model is the only component evaluated against real outcomes when trained on Land Registry data; it is observational."]
    if ev.get("holdout"):
        h = ev["holdout"]
        L += ["", "## Held-out test predictions (full engine; files in data/predictions/)", "",
              f"{h['n']} test homes run through the whole engine with their observed (noisy, partly missing) usage. "
              f"Full-package savings vs simulator reference: electricity MAE {h['pkg_elec_mae']:,.0f} kWh, gas MAE {h['pkg_gas_mae']:,.0f} kWh "
              f"(baseline: fixed % of baseline gas, MAE {h['pkg_gas_mae_fixed_pct']:,.0f}). Improvement-potential ranking Spearman "
              f"{h['ip_spearman']:.3f} (EPC-band order {h['ip_spearman_epc_band']:.3f}). Poor/Very poor flagged: {h['share_poor']:.0%}. "
              f"Abstained: {h['abstained']}."]
    if ev.get("llm"):
        l = ev["llm"]
        L += ["", "## LLM layer", "", f"Questions {l.get('n')}; verifier pass rate {l.get('verifier_pass_rate')}; template fallback used {l.get('fallback_rate')}. "
              + ("Manual audit: groundedness {groundedness}, factual accuracy {factual_accuracy}, unsupported claims per sentence {unsupported_claims_per_sentence} over {audited_answers} answers.".format(**l["manual_audit"])
               if l.get("manual_audit") else "Groundedness and factual accuracy: fill the MANUAL_* columns of docs/llm_audit.csv and run `make llm-audit`.")]
    L += ["", "## Reporting rules", "", "- Beating the simple baseline shows usefulness as a surrogate, not truth.",
          "- Dashboard wording: estimated by a simulation model, not measured; savings are estimates, not guarantees."]
    (ROOT / "docs" / "evaluation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(planner_homes: int = 60, shift_homes: int = 1500) -> dict:
    props, lab, X = load_dataset()
    with open(MODELS_DIR / "metrics.json", encoding="utf-8") as fh:
        training = json.load(fh)
    ev = {"generated": time.strftime("%Y-%m-%d %H:%M"), "training_metrics": training}
    print("shift test ...", flush=True)
    ev["shift_test"] = shift_test(lab, X, props, shift_homes)
    print("planner eval ...", flush=True)
    ev["planner"] = planner_eval(lab, props, planner_homes)
    ev["score_checks"] = score_checks()
    for name, key in (("value_metrics.json", "value"), ("llm_eval.json", "llm"), ("holdout_metrics.json", "holdout")):
        path = MODELS_DIR / name
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                ev[key] = json.load(fh)
    with open(MODELS_DIR / "evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(ev, fh, indent=2, default=float)
    write_report(ev)
    print("wrote docs/evaluation_report.md")
    return ev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-homes", type=int, default=60)
    ap.add_argument("--shift-homes", type=int, default=1500)
    a = ap.parse_args()
    main(a.planner_homes, a.shift_homes)
