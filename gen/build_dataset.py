"""Build the synthetic dataset (spec 6): sample -> simulate labels -> noise and missing
data -> customer form view -> grouped splits -> Parquet tables.

    python -m gen.build_dataset --n 30000 --seed 42 [--calibrate]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from engine.baseline import annualise
from gen.labels import build_retrofit_reference, label_many
from gen.noise import missing_months, observe
from gen.sampling import form_view, sample_population
from gen.splits import assign_splits, check_no_group_leak
from sim.home import HomeState
from sim.parameters import PROCESSED_DIR, electricity_shape, gas_shape, reload


def build(n: int = 30000, seed: int = 42, calibrate: bool = False, workers: int | None = None) -> dict:
    t0 = time.time()
    if calibrate:
        from gen.calibration import run_calibration

        print("calibrating against NEED targets ...", flush=True)
        run_calibration()
        reload()
    print(f"sampling {n} properties ...", flush=True)
    pop = sample_population(n, seed)
    rows = pop.to_dict("records")
    print("simulating labels ...", flush=True)
    labels = pd.DataFrame(label_many(rows, workers))
    print(f"  labels done in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(seed + 1)
    usage_rows, obs_rows, forms = [], [], []
    for row, lab in zip(rows, labels.to_dict("records")):
        state = HomeState.from_dict(json.loads(row["state_json"]))
        drop = missing_months(rng)
        elec_clean = np.asarray(json.loads(lab["monthly_electricity_clean"]))
        gas_clean = np.asarray(json.loads(lab["monthly_gas_clean"]))
        elec_obs = observe(elec_clean, rng, drop)
        gas_obs = observe(gas_clean, rng, drop) if gas_clean.sum() > 0 else [0.0] * 12
        if gas_clean.sum() <= 0:
            gas_obs = [None if m in drop else 0.0 for m in range(12)]
        for m in range(12):
            usage_rows.append({"property_id": row["property_id"], "month": m + 1, "electricity_kwh": elec_obs[m], "gas_kwh": gas_obs[m], "missing_flag": m in drop})
        obs_elec, _ = annualise(elec_obs, electricity_shape())
        obs_gas = annualise(gas_obs, gas_shape(state.region))[0] if gas_clean.sum() > 0 else 0.0
        obs_rows.append({"property_id": row["property_id"], "obs_electricity_kwh": obs_elec, "obs_gas_kwh": obs_gas, "obs_months_present": 12 - len(drop),
                         "obs_monthly_electricity": json.dumps(elec_obs), "obs_monthly_gas": json.dumps(gas_obs)})
        forms.append(json.dumps(form_view(state, row["postcode"], rng)))

    props = pop[["property_id", "customer_id", "postcode", "state_json"]].copy()
    props["form_json"] = forms
    st = pd.DataFrame([json.loads(s) for s in props.state_json])
    for col in ("region", "property_type", "age_band", "heating", "floor_area", "tenure"):
        props[col] = st[col].values
    props["split"] = assign_splits(props, seed)
    check_no_group_leak(props)

    hidden = pop[["property_id"] + [c for c in pop.columns if c.startswith("hidden_")]]
    usage = pd.DataFrame(usage_rows)
    obs = pd.DataFrame(obs_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    props.to_parquet(PROCESSED_DIR / "properties.parquet", index=False)
    hidden.to_parquet(PROCESSED_DIR / "hidden_state.parquet", index=False)
    usage.to_parquet(PROCESSED_DIR / "energy_usage.parquet", index=False)
    labels.merge(obs, on="property_id").to_parquet(PROCESSED_DIR / "labels.parquet", index=False)

    train_ids = set(props.loc[props.split == "train", "property_id"])
    train_labels = [l for l in labels.to_dict("records") if l["property_id"] in train_ids]
    ref = build_retrofit_reference(labels=train_labels)
    with open(PROCESSED_DIR / "retrofit_reference.json", "w", encoding="utf-8") as fh:
        json.dump(ref, fh)

    summary = {
        "n": n, "seed": seed, "seconds": round(time.time() - t0, 1),
        "splits": props.split.value_counts().to_dict(),
        "customers": int(props.customer_id.nunique()),
        "share_missing_months": float(usage.missing_flag.mean()),
    }
    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    build(a.n, a.seed, a.calibrate, a.workers)
