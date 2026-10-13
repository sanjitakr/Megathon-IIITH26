"""Held-out test predictions (submission item I) and the improvement-potential ranking.

Runs every test-split home through the full engine with the usage a customer would supply
(noisy, partly missing months) and writes:
  data/predictions/test_predictions.csv    one row per home: engine outputs + reference labels
  data/predictions/portfolio_ranking.csv   homes ranked by improvement potential
  data/models/holdout_metrics.json         savings vs simulator reference, ranking quality
Column meanings: docs/data_schema.md.

    python -m ml.predict_holdout [--split test] [--limit 0]
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from engine.inputs import CustomerInput, Usage
from sim.parameters import MODELS_DIR, PROCESSED_DIR, ROOT

PRED_DIR = ROOT / "data" / "predictions"


def _one(args) -> dict:
    from engine.pipeline import analyse

    pid, form_json, elec, gas = args
    form = json.loads(form_json)
    form["electricity"] = Usage(monthly_kwh=json.loads(elec))
    g = json.loads(gas)
    if any(x for x in g if x):
        form["gas"] = Usage(monthly_kwh=g)
    res = analyse(CustomerInput(**form), with_ranges=False)
    row = {"property_id": pid, "status": res["status"]}
    if res["status"] != "ok":
        return row
    cur, pk, best = res["score"]["current"], res["retrofits"]["package_row"], res["plan"].get("best") or {}
    row.update({
        "tier": res["confidence"]["tier"],
        "pred_baseline_electricity_kwh": res["baseline"]["purchased_kwh"]["electricity"],
        "pred_baseline_gas_kwh": res["baseline"]["purchased_kwh"]["gas"],
        "pred_score": cur["score_rounded"], "pred_class": cur["class"], "pred_is_poor": res["efficiency_flag"]["is_poor"],
        "pred_key_issues": "; ".join(i["issue"] for i in res["efficiency_flag"]["key_issues"]),
        "pred_band": res["rating"]["recorded_band"] or res["rating"]["simulated"]["band"],
        "pred_improvement_potential_pct": res["efficiency_flag"]["improvement_potential"].get("pct"),
        "pred_feasible_retrofits": ";".join(r["retrofit"] for r in res["retrofits"]["feasible_list"]),
        "pred_pkg_d_elec_kwh": pk.get("delta_kwh", {}).get("electricity"),
        "pred_pkg_d_gas_kwh": pk.get("delta_kwh", {}).get("gas"),
        "pred_pkg_carbon_saving_kg": pk.get("carbon_saving_kg"),
        "pred_recommended": ";".join(best.get("retrofits", [])),
        "pred_recommended_d_elec_kwh": (best.get("delta_kwh") or {}).get("electricity"),
        "pred_recommended_d_gas_kwh": (best.get("delta_kwh") or {}).get("gas"),
        "pred_recommended_carbon_kg": best.get("carbon_saving_kg"),
    })
    return row


def main(split: str = "test", limit: int = 0) -> dict:
    props = pd.read_parquet(PROCESSED_DIR / "properties.parquet")
    lab = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    df = props[props.split == split].merge(lab, on="property_id")
    if limit:
        df = df.head(limit)
    args = list(zip(df.property_id, df.form_json, df.obs_monthly_electricity, df.obs_monthly_gas))
    with get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = pool.map(_one, args, chunksize=20)
    pred = pd.DataFrame(rows)
    ref = df[["property_id", "region", "property_type", "age_band", "clean_electricity_kwh", "clean_gas_kwh", "pkg_d_elec", "pkg_d_gas",
              "improvement_potential_pct", "epc_band_true", "sap_points"]].rename(columns={
        "clean_electricity_kwh": "ref_baseline_electricity_kwh", "clean_gas_kwh": "ref_baseline_gas_kwh", "pkg_d_elec": "ref_pkg_d_elec_kwh",
        "pkg_d_gas": "ref_pkg_d_gas_kwh", "improvement_potential_pct": "ref_improvement_potential_pct", "epc_band_true": "ref_band",
        "sap_points": "ref_sap_points"})
    out = ref.merge(pred, on="property_id")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PRED_DIR / f"{split}_predictions.csv", index=False)

    ok = out[out.status == "ok"].copy()
    rank = ok.sort_values("pred_improvement_potential_pct", ascending=False)[
        ["property_id", "region", "property_type", "age_band", "pred_score", "pred_class", "pred_is_poor", "pred_band",
         "pred_improvement_potential_pct", "pred_key_issues", "pred_recommended"]]
    rank.insert(0, "rank", range(1, len(rank) + 1))
    rank.to_csv(PRED_DIR / "portfolio_ranking.csv", index=False)

    both = ok.dropna(subset=["pred_improvement_potential_pct", "ref_improvement_potential_pct"])
    band_order = both.pred_band.map({b: i for i, b in enumerate("ABCDEFG")})
    gas_ratio = np.nanmean(ok.ref_pkg_d_gas_kwh / ok.ref_baseline_gas_kwh.replace(0, np.nan))
    metrics = {
        "split": split, "n": int(len(out)), "abstained": int((out.status != "ok").sum()),
        "pkg_elec_mae": float(np.mean(np.abs(ok.pred_pkg_d_elec_kwh - ok.ref_pkg_d_elec_kwh))),
        "pkg_gas_mae": float(np.mean(np.abs(ok.pred_pkg_d_gas_kwh - ok.ref_pkg_d_gas_kwh))),
        "pkg_gas_mae_fixed_pct": float(np.mean(np.abs(gas_ratio * ok.pred_baseline_gas_kwh - ok.ref_pkg_d_gas_kwh))),
        "ip_spearman": float(spearmanr(both.pred_improvement_potential_pct, both.ref_improvement_potential_pct).correlation),
        "ip_spearman_epc_band": float(spearmanr(band_order, both.ref_improvement_potential_pct).correlation),
        "share_poor": float(ok.pred_is_poor.mean()),
        "baseline_elec_mape": float(np.mean(np.abs(ok.pred_baseline_electricity_kwh - ok.ref_baseline_electricity_kwh)
                                            / ok.ref_baseline_electricity_kwh) * 100),
    }
    (MODELS_DIR / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.split, a.limit)
