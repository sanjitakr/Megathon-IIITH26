"""LLM evaluation (spec 11): held-out questions incl. missing data, conflicts and infeasible
retrofits. Reports verifier pass rate and fallback rate automatically; groundedness and
factual accuracy need a human: fill the three MANUAL_* columns (last three) of
docs/llm_audit.csv, then score them (targets: 90%, 90%, under 5% unsupported claims).

    python -m llm.evaluate [--limit 10]     # run questions, write docs/llm_audit.csv
    python -m llm.evaluate --score-audit    # after filling the MANUAL_* columns
"""
from __future__ import annotations

import argparse
import csv
import json
import time

from engine.inputs import CustomerInput
from engine.whatif import run_whatif
from engine.pipeline import analyse
from llm.explainer import explain
from llm.parser import parse
from llm.providers import LLMChain, status
from sim.parameters import MODELS_DIR, ROOT

QUESTIONS_PATH = ROOT / "llm" / "eval_questions.json"


def run(limit: int | None = None) -> dict:
    spec = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    profiles = spec["profiles"]
    items = spec["questions"][:limit] if limit else spec["questions"]
    st = status()
    if not st["available"]:
        print("WARNING: no LLM provider is available, every answer will be a template. "
              "Check LLM_PROVIDER and the API key in .env.", flush=True)
    chain = LLMChain()
    rows, passed, fallback, unsupported_total, numbers_total = [], 0, 0, 0, 0
    for i, q in enumerate(items):
        print(f"[{i + 1}/{len(items)}] {q['question']}", flush=True)
        inp = CustomerInput(**profiles[q["profile"]])
        sc, parsed_by = parse(q["question"], chain)
        if sc.is_empty():
            res, diff, req = analyse(inp, with_ranges=False), None, []
        else:
            w = run_whatif(inp, sc, with_ranges=False)
            res, diff, req = w["results"], w["diff"], w["requested_infeasible"]
        ex = explain(res, q["question"], requested=req, diff=diff, question_type=sc.question_type, chain=chain)
        from llm.verifier import verify

        v = verify(ex["text"], ex["facts"], q["question"])
        passed += int(v["ok"])
        fallback += int(ex["source"] == "template")
        unsupported_total += len(v["unsupported"])
        numbers_total += v["n_numbers"]
        llm_errors = "; ".join(ex.get("errors", []) + [f"attempt {k + 1} unsupported: {', '.join(a['unsupported'])}" for k, a in enumerate(ex["attempts"]) if not a["ok"]])
        gem = next((p for p in chain.providers if p.name == "gemini"), None)
        rows.append({"id": i, "category": q["category"], "profile": q["profile"], "question": q["question"],
                     "parsed_by": parsed_by, "scenario": json.dumps(sc.model_dump(exclude_defaults=True)),
                     "answer_source": ex["source"], "model": gem.model if (gem and ex["source"] == "gemini") else "",
                     "verifier_ok": v["ok"], "unsupported": "; ".join(v["unsupported"]), "llm_errors": llm_errors[:500],
                     "answer": ex["text"],
                     "MANUAL_grounded (Y/N)": "", "MANUAL_factually_accurate (Y/N)": "", "MANUAL_unsupported_claims (count)": ""})
    n = len(rows)
    out = {
        "n": n,
        "llm_status": status(),
        "verifier_pass_rate": round(passed / n, 3),
        "fallback_rate": round(fallback / n, 3),
        "unsupported_number_rate": round(unsupported_total / max(numbers_total, 1), 4),
        "by_category": {},
        "note": ("All answers came from templates: no LLM answered (see the llm_errors column)." if fallback == len(rows)
                 else "Fill the MANUAL_* columns in docs/llm_audit.csv, then run python -m llm.evaluate --score-audit."),
    }
    for r in rows:
        c = out["by_category"].setdefault(r["category"], {"n": 0, "verifier_ok": 0})
        c["n"] += 1
        c["verifier_ok"] += int(r["verifier_ok"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "llm_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(ROOT / "docs" / "llm_audit.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps({k: v for k, v in out.items() if k != "by_category"}, indent=2))
    if fallback:
        first = next((r["llm_errors"] for r in rows if r["llm_errors"]), "")
        print(f"{fallback}/{n} answers fell back to templates. First LLM error: {first[:300]}")
    return out


def score_audit() -> dict:
    """Groundedness, factual accuracy and unsupported-claim rate from the filled audit sheet."""
    path = ROOT / "docs" / "llm_audit.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    llm_rows = [r for r in rows if r["answer_source"] != "template"]

    def yes(r, col):
        return r[col].strip().upper().startswith("Y")

    filled = [r for r in llm_rows if r["MANUAL_grounded (Y/N)"].strip() and r["MANUAL_factually_accurate (Y/N)"].strip()]
    if not filled:
        raise SystemExit("No LLM-written answers have the MANUAL_* columns filled yet (template answers are excluded).")
    claims = sum(int(r["MANUAL_unsupported_claims (count)"] or 0) for r in filled)
    sentences = sum(max(1, r["answer"].count(". ") + 1) for r in filled)
    out = {
        "audited_answers": len(filled),
        "groundedness": round(sum(yes(r, "MANUAL_grounded (Y/N)") for r in filled) / len(filled), 3),
        "factual_accuracy": round(sum(yes(r, "MANUAL_factually_accurate (Y/N)") for r in filled) / len(filled), 3),
        "unsupported_claims_per_sentence": round(claims / sentences, 4),
        "targets": {"groundedness": 0.9, "factual_accuracy": 0.9, "unsupported_claims_per_sentence": "< 0.05"},
    }
    ev_path = MODELS_DIR / "llm_eval.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {}
    ev["manual_audit"] = out
    ev_path.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-audit", action="store_true", help="score the filled MANUAL_* columns")
    a = ap.parse_args()
    score_audit() if a.score_audit else run(a.limit)
