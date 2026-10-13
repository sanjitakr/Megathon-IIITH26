"""Hedonic property value model (spec 10).

log(price) = f(floor area, type, region, age band, EPC band, sale date), LightGBM with a
monotone constraint on EPC band, plus bootstrap models for the uplift range and quantile
models for the price interval. Uplift = predicted value at the post-retrofit band minus at
the current band, everything else fixed: indicative and observational, not causal.

Real data: a CSV of HM Land Registry Price Paid joined to EPC (data/raw/land_registry_epc.csv)
with columns price, date, floor_area, property_type, region, age_band, epc_band.

    python -m ml.value_model --csv data/raw/land_registry_epc.csv
    python -m ml.value_model --synthetic-demo     # pipeline test only, flagged DEMO ONLY
"""
from __future__ import annotations

import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.conformal import coverage
from sim.parameters import MODELS_DIR, all_regions

BANDS = list("ABCDEFG")
TYPES = ["detached", "semi", "terraced", "end_terrace", "flat", "bungalow"]
AGES = list("ABCDEFGHIJK")
FEATS = ["floor_area", "property_type", "region", "age_band", "epc_rank", "sale_year"]
MIN_ROWS = 2000
THIN_CELL = 30


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "floor_area": df.floor_area.astype(float),
        "property_type": pd.Categorical(df.property_type, categories=TYPES),
        "region": pd.Categorical(df.region, categories=all_regions()),
        "age_band": pd.Categorical(df.age_band, categories=AGES),
        "epc_rank": df.epc_band.map({b: 7 - i for i, b in enumerate(BANDS)}).astype(float),  # A=7 ... G=1
        "sale_year": pd.to_datetime(df.date).dt.year.astype(float) if "date" in df else df.sale_year.astype(float),
    })
    return out


def synthetic_sales(n: int = 20000, seed: int = 5) -> pd.DataFrame:
    """Fake sales for testing the pipeline only. Never report these as evidence."""
    rng = np.random.default_rng(seed)
    regions = all_regions()
    base = {r: v for r, v in zip(regions, [520, 400, 330, 360, 260, 260, 230, 230, 220, 170, 210, 190])}
    tmult = {"detached": 1.45, "semi": 1.0, "terraced": 0.85, "end_terrace": 0.9, "flat": 0.75, "bungalow": 1.1}
    rows = []
    for _ in range(n):
        r = regions[rng.integers(len(regions))]
        t = TYPES[rng.integers(len(TYPES))]
        area = float(np.clip(rng.lognormal(np.log(85), 0.3), 30, 400))
        band = BANDS[min(6, max(0, int(rng.normal(3, 1))))]
        year = int(rng.integers(2016, 2026))
        price = base[r] * 1000 * tmult[t] * (area / 85) ** 0.75 * (1 + 0.025 * (3 - BANDS.index(band))) * (1.03 ** (year - 2016))
        price *= np.exp(rng.normal(0, 0.18))
        rows.append({"price": price, "date": f"{year}-06-01", "floor_area": area, "property_type": t, "region": r,
                     "age_band": AGES[rng.integers(len(AGES))], "epc_band": band})
    return pd.DataFrame(rows)


def train(df: pd.DataFrame, basis: str, seed: int = 0) -> dict:
    df = df.dropna(subset=["price", "floor_area", "property_type", "region", "epc_band"]).reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise SystemExit(f"Only {len(df)} joined sales: too thin to fit a value model (need {MIN_ROWS}). Abstaining.")
    X = prepare(df)
    y = np.log(df.price.values)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_te = len(df) // 5
    te, tr = idx[:n_te], idx[n_te:]
    mono = [1 if f == "epc_rank" else 0 for f in FEATS]
    params = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=30, monotone_constraints=mono, verbose=-1)
    median = lgb.LGBMRegressor(**params, random_state=seed).fit(X.iloc[tr], y[tr])
    qparams = {k: v for k, v in params.items() if k != "monotone_constraints"}  # not supported for quantile loss
    q10 = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **qparams).fit(X.iloc[tr], y[tr])
    q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **qparams).fit(X.iloc[tr], y[tr])
    boots = []
    for b in range(8):
        bi = rng.choice(tr, size=len(tr), replace=True)
        boots.append(lgb.LGBMRegressor(**params, random_state=b).fit(X.iloc[bi], y[bi]))
    pred = np.exp(median.predict(X.iloc[te]))
    true = df.price.values[te]
    cell_mean = df.iloc[tr].groupby(["region", "property_type"]).price.mean()
    base = np.array([cell_mean.get((r, t), df.price.iloc[tr].mean()) for r, t in zip(df.region.iloc[te], df.property_type.iloc[te])])
    cell_counts = df.groupby(["region", "property_type"]).size()
    meta = {
        "basis": basis,
        "n_sales": int(len(df)),
        "thin_join": bool((cell_counts < THIN_CELL).mean() > 0.25),
        "evidence": {"n_sales": int(len(df)), "years": [int(X.sale_year.min()), int(X.sale_year.max())]},
    }
    metrics = {
        "basis": basis,
        "mape_pct": round(float(np.mean(np.abs(pred - true) / true) * 100), 2),
        "coverage_80": round(coverage(np.log(true), q10.predict(X.iloc[te]), q90.predict(X.iloc[te])), 3),
        "baseline_mape_pct": round(float(np.mean(np.abs(base - true) / true) * 100), 2),
        "n_test": int(n_te),
    }
    return {"median": median, "q10": q10, "q90": q90, "boots": boots, "meta": meta, "metrics": metrics}


def predict_uplift(bundle: dict, state, current_band: str, after_band: str, sale_year: int = 2026) -> dict:
    row = pd.DataFrame([{"price": 1, "date": f"{sale_year}-06-01", "floor_area": state.floor_area, "property_type": state.property_type,
                         "region": state.region, "age_band": state.age_band, "epc_band": current_band}])
    Xc = prepare(row)
    Xa = Xc.copy()
    Xa["epc_rank"] = 7 - BANDS.index(after_band)
    ups = [100 * (np.exp(m.predict(Xa)[0] - m.predict(Xc)[0]) - 1) for m in [bundle["median"]] + bundle["boots"]]
    p10, p50, p90 = np.percentile(ups, [10, 50, 90])
    if bundle["meta"].get("thin_join"):
        spread = max(p90 - p10, 1.0)
        p10, p90 = p10 - spread / 2, p90 + spread / 2
    return {"p10": float(p10), "p50": float(ups[0]), "p90": float(p90)}


def main(csv: str | None, synthetic_demo: bool) -> None:
    if csv:
        df = pd.read_csv(csv)
        bundle = train(df, "land_registry_epc")
    elif synthetic_demo:
        bundle = train(synthetic_sales(), "synthetic_demo")
    else:
        raise SystemExit("Give --csv (Land Registry joined to EPC) or --synthetic-demo")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({k: bundle[k] for k in ("median", "q10", "q90", "boots", "meta")}, MODELS_DIR / "value_model.joblib")
    with open(MODELS_DIR / "value_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(bundle["metrics"], fh, indent=2)
    print(json.dumps(bundle["metrics"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--synthetic-demo", action="store_true")
    a = ap.parse_args()
    main(a.csv, a.synthetic_demo)
