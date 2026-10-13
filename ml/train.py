"""Train every ML component (spec 7) on the synthetic dataset.

    python -m ml.train [--tune]

Baseline electricity: LightGBM on log(observed) - log(physics estimate) (residual learning).
Baseline gas: classifier (has gas) then LightGBM residual model on positives.
Intervals: split-conformal CQR, 80% nominal. Feasibility: LightGBM on rule labels for
incomplete inputs. Per-retrofit saving surrogates (live preview only; the simulator stays
the validator). Ranking: LambdaRank vs a heuristic. Improvement potential: regression.
Metrics are agreement with the reference simulator, never real-world accuracy.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, ndcg_score, precision_score, r2_score, recall_score

from ml.conformal import CQR, coverage
from ml.features import CAT_COLS, FEATURES, FEATURES_NO_PHYS, build_feature_frame
from sim.parameters import MODELS_DIR, PROCESSED_DIR, retrofit_ids

SEED = 42
BASE_PARAMS = {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 60,
               "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.85, "reg_lambda": 1.0,
               "random_state": SEED, "verbose": -1}


# ------------------------------------------------------------------ data

def load_dataset(rebuild_features: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    props = pd.read_parquet(PROCESSED_DIR / "properties.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    feat_path = PROCESSED_DIR / "features.parquet"
    if feat_path.exists() and not rebuild_features:
        X = pd.read_parquet(feat_path)
        X = _restore_categories(X)
    else:
        X = build_feature_frame(props.form_json.tolist())
        X.to_parquet(feat_path, index=False)
    labels = props[["property_id", "split", "region", "property_type", "age_band", "customer_id"]].merge(labels, on="property_id")
    return props, labels.reset_index(drop=True), X.reset_index(drop=True)


def _restore_categories(X: pd.DataFrame) -> pd.DataFrame:
    from ml.features import to_frame

    return to_frame(X.to_dict("records"))


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    mask = y > 100
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs(pred[mask] - y[mask]) / y[mask]) * 100) if mask.any() else None,
        "r2": float(r2_score(y, pred)),
        "n": int(len(y)),
    }


def monotone_vector(features: list[str], positive: tuple[str, ...]) -> list[int]:
    return [1 if f in positive else 0 for f in features]


def maybe_tune(X, y, Xv, yv, n_trials: int) -> dict:
    try:
        import optuna
    except ImportError:
        print("optuna not installed: using default hyper-parameters")
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            **BASE_PARAMS,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        m = lgb.LGBMRegressor(**params).fit(X, y)
        return float(np.mean(np.abs(m.predict(Xv) - yv)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# ------------------------------------------------------------------ energy

def fit_energy(X, lab, tune: bool = False) -> tuple[dict, dict]:
    tr, va, te, ood = (lab.split == s for s in ("train", "val", "test", "ood_test"))
    out_models, report = {}, {}

    # Electricity: residual learning on log scale
    y_e = np.log(lab.obs_electricity_kwh.clip(lower=50).values)
    phys_e = np.log(X.phys_elec.clip(lower=50).values)
    r_e = y_e - phys_e
    params = dict(BASE_PARAMS)
    if tune:
        params.update(maybe_tune(X[tr], r_e[tr], X[va], r_e[va], 60))
    elec = lgb.LGBMRegressor(**params).fit(X[tr], r_e[tr])
    cqr_e = CQR(0.2).fit(X[tr], r_e[tr]).calibrate(X[va], r_e[va])
    out_models["electricity"] = {"model": elec, "cqr": cqr_e, "params": params}

    def predict_e(mask):
        p50 = np.exp(phys_e[mask] + elec.predict(X[mask]))
        lo, hi = cqr_e.predict(X[mask])
        return p50, np.exp(phys_e[mask] + lo), np.exp(phys_e[mask] + hi)

    # Gas: has-gas classifier + residual model on positives
    has_gas = (lab.clean_gas_kwh > 0).astype(int).values
    clf = lgb.LGBMClassifier(**{**BASE_PARAMS, "n_estimators": 300}).fit(X[tr], has_gas[tr])
    pos = has_gas == 1
    y_g = np.log1p(lab.obs_gas_kwh.clip(lower=0).values)
    phys_g = np.log1p(X.phys_gas.values)
    r_g = y_g - phys_g
    gas = lgb.LGBMRegressor(**params).fit(X[tr & pos], r_g[tr & pos])
    cqr_g = CQR(0.2).fit(X[tr & pos], r_g[tr & pos]).calibrate(X[va & pos], r_g[va & pos])
    out_models["gas"] = {"classifier": clf, "model": gas, "cqr": cqr_g, "params": params}

    def predict_g(mask):
        prob = clf.predict_proba(X[mask])[:, 1]
        p50 = np.expm1(phys_g[mask] + gas.predict(X[mask]))
        lo, hi = cqr_g.predict(X[mask])
        on = prob >= 0.5
        return np.where(on, p50, 0.0), np.where(on, np.expm1(phys_g[mask] + lo), 0.0), np.where(on, np.expm1(phys_g[mask] + hi), 0.0), prob

    # Simple baseline: mean by (type, age band) from train
    def cell_mean(target):
        means = lab[tr].groupby(["property_type", "age_band"])[target].mean()
        overall = lab[tr][target].mean()
        return lab.apply(lambda r: means.get((r.property_type, r.age_band), overall), axis=1).values

    base_e = cell_mean("obs_electricity_kwh")
    base_g = cell_mean("obs_gas_kwh")

    # Ablations (direct log target with/without physics features)
    abl = {}
    mono = monotone_vector(FEATURES, ("floor_area", "occupants", "bedrooms"))
    direct = lgb.LGBMRegressor(**{**params, "monotone_constraints": mono}).fit(X[tr], y_e[tr])
    abl["electricity_physics_no_residual"] = metrics(np.exp(y_e[te]), np.exp(direct.predict(X[te])))
    mono_np = monotone_vector(FEATURES_NO_PHYS, ("floor_area", "occupants", "bedrooms"))
    nophys = lgb.LGBMRegressor(**{**params, "monotone_constraints": mono_np}).fit(X.loc[tr, FEATURES_NO_PHYS], y_e[tr])
    abl["electricity_no_physics_no_residual"] = metrics(np.exp(y_e[te]), np.exp(nophys.predict(X.loc[te, FEATURES_NO_PHYS])))
    abl["electricity_physics_only"] = metrics(np.exp(y_e[te]), np.exp(phys_e[te]))
    g_direct = lgb.LGBMRegressor(**params).fit(X.loc[tr & pos, FEATURES_NO_PHYS], y_g[tr & pos])
    abl["gas_no_physics_no_residual"] = metrics(np.expm1(y_g[te & pos]), np.expm1(g_direct.predict(X.loc[te & pos, FEATURES_NO_PHYS])))
    abl["gas_physics_only"] = metrics(np.expm1(y_g[te & pos]), np.expm1(phys_g[te & pos]))

    # Optional small ensemble (CatBoost / XGBoost) if installed
    ens = {}
    for name in ("catboost", "xgboost"):
        try:
            if name == "catboost":
                from catboost import CatBoostRegressor

                m = CatBoostRegressor(iterations=600, depth=6, verbose=0, random_seed=SEED, cat_features=CAT_COLS)
                Xc = X.copy()
                for c in CAT_COLS:
                    Xc[c] = Xc[c].astype(str)
                m.fit(Xc[tr], r_e[tr])
                ens[name] = metrics(np.exp(y_e[te]), np.exp(phys_e[te] + m.predict(Xc[te])))
            else:
                import xgboost as xgb

                m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=7, enable_categorical=True, tree_method="hist", random_state=SEED)
                m.fit(X[tr], r_e[tr])
                ens[name] = metrics(np.exp(y_e[te]), np.exp(phys_e[te] + m.predict(X[te])))
        except ImportError:
            ens[name] = "not installed"

    for split_name, mask in (("val", va), ("test", te), ("ood_test", ood)):
        p50, lo, hi = predict_e(mask)
        y = lab.obs_electricity_kwh.values[mask]
        report[f"electricity_{split_name}"] = {**metrics(y, p50), "coverage_80": coverage(y, lo, hi), "mean_width": float(np.mean(hi - lo)),
                                               "baseline_type_age_mean": metrics(y, base_e[mask])}
        gp50, glo, ghi, prob = predict_g(mask)
        y = lab.obs_gas_kwh.values[mask]
        has = pos[mask]
        report[f"gas_{split_name}"] = {**metrics(y[has], gp50[has]), "coverage_80": coverage(y[has], glo[has], ghi[has]),
                                       "mean_width": float(np.mean(ghi[has] - glo[has])),
                                       "has_gas_accuracy": float(np.mean((prob >= 0.5) == has)),
                                       "baseline_type_age_mean": metrics(y[has], base_g[mask][has])}
    report["ablations_test"] = abl
    report["ensemble_members_test"] = ens

    # Fairness: errors by region, type, age band (test + ood combined)
    mask = te | ood
    p50, _, _ = predict_e(mask)
    sub = lab[mask].assign(pred=p50, err=np.abs(p50 - lab.obs_electricity_kwh.values[mask]) / lab.obs_electricity_kwh.values[mask] * 100)
    report["fairness_electricity_mape"] = {
        g: sub.groupby(g)["err"].mean().round(2).to_dict() for g in ("region", "property_type", "age_band")
    }
    out_models["_predict_fns"] = None
    return out_models, report


# ------------------------------------------------------------------ feasibility

def fit_feasibility(X, lab, props) -> tuple[dict, dict]:
    from engine.feasibility import assess
    from engine.inputs import CustomerInput
    from engine.state_builder import build_state

    tr, te = lab.split == "train", lab.split == "test"
    models, rep = {}, {}
    # rule engine on defaults-filled inputs (what the engine does at runtime)
    test_forms = props.set_index("property_id").loc[lab.property_id[te], "form_json"].tolist()
    rule_pred = []
    for f in test_forms:
        st, _ = build_state(CustomerInput(**json.loads(f)))
        rule_pred.append(set(assess(st).feasible))
    for rid in retrofit_ids():
        y = lab[f"feasible_{rid}"].values
        if y[tr].min() == y[tr].max():
            continue
        clf = lgb.LGBMClassifier(**{**BASE_PARAMS, "n_estimators": 300}).fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        rules = np.array([rid in s for s in rule_pred], dtype=int)
        majority = np.full_like(y[te], int(y[tr].mean() >= 0.5))
        rep[rid] = {
            "classifier": {"precision": float(precision_score(y[te], pred, zero_division=0)), "recall": float(recall_score(y[te], pred, zero_division=0))},
            "rules_on_filled_inputs": {"precision": float(precision_score(y[te], rules, zero_division=0)), "recall": float(recall_score(y[te], rules, zero_division=0))},
            "majority": {"precision": float(precision_score(y[te], majority, zero_division=0)), "recall": float(recall_score(y[te], majority, zero_division=0))},
            "prevalence": float(y[te].mean()),
        }
        models[rid] = clf
    return models, rep


# ------------------------------------------------------------------ savings surrogates

def fit_surrogates(X, lab) -> tuple[dict, dict]:
    tr, te = lab.split == "train", lab.split == "test"
    models, rep = {}, {}
    for rid in retrofit_ids():
        feas = lab[f"feasible_{rid}"].values == 1
        if (tr & feas).sum() < 200:
            continue
        models[rid] = {}
        rep[rid] = {}
        for key in ("d_elec", "d_gas"):
            y = lab[f"{key}_{rid}"].values
            m = lgb.LGBMRegressor(**{**BASE_PARAMS, "n_estimators": 400}).fit(X[tr & feas], y[tr & feas])
            pred = m.predict(X[te & feas])
            models[rid][key] = m
            # baseline: brief-style fixed percentage of the relevant baseline fuel (mean ratio on train)
            base_col = "clean_electricity_kwh" if key == "d_elec" else "clean_gas_kwh"
            denom = lab[base_col].values
            ratio = np.nanmean(np.divide(y[tr & feas], denom[tr & feas], out=np.zeros((tr & feas).sum()), where=denom[tr & feas] > 0))
            rep[rid][key] = {"surrogate_mae": float(mean_absolute_error(y[te & feas], pred)),
                             "fixed_pct_baseline_mae": float(mean_absolute_error(y[te & feas], ratio * denom[te & feas])),
                             "n_test": int((te & feas).sum())}
    return models, rep


# ------------------------------------------------------------------ ranking

def fit_ranker(X, lab) -> tuple[object, dict]:
    ids = retrofit_ids()
    rows, groups, rel, split = [], [], [], []
    for i, r in lab.iterrows():
        items = [rid for rid in ids if r[f"feasible_{rid}"] == 1 and np.isfinite(r[f"saving_per_gbp_{rid}"])]
        if len(items) < 2:
            continue
        vals = np.array([r[f"saving_per_gbp_{rid}"] for rid in items])
        order = vals.argsort().argsort()
        grades = np.floor(4 * order / max(len(items) - 1, 1)).astype(int)
        for rid, g in zip(items, grades):
            rows.append((i, rid))
            rel.append(int(g))
        groups.append(len(items))
        split.append(r.split)
    base = X.loc[[i for i, _ in rows]].reset_index(drop=True)
    for rid in ids:
        base[f"is_{rid}"] = [int(r == rid) for _, r in rows]
    rel = np.array(rel)
    row_split = np.repeat(split, groups)
    gidx = np.repeat(np.arange(len(groups)), groups)
    tr = row_split == "train"
    te = row_split == "test"
    tr_groups = np.bincount(gidx[tr])
    tr_groups = tr_groups[tr_groups > 0]
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)
    ranker.fit(base[tr], rel[tr], group=tr_groups)
    # heuristic: mean saving per GBP of each retrofit on train
    item_mean = {rid: float(np.nanmean(lab.loc[lab.split == "train", f"saving_per_gbp_{rid}"])) for rid in ids}
    nd_r, nd_h = [], []
    score = ranker.predict(base[te])
    te_rows = np.where(te)[0]
    for g in np.unique(gidx[te]):
        sel = gidx[te_rows] == g
        idx = te_rows[sel]
        y_true = rel[idx][None, :]
        nd_r.append(ndcg_score(y_true, score[sel][None, :], k=3))
        heur = np.array([item_mean[rows[j][1]] for j in idx])[None, :]
        nd_h.append(ndcg_score(y_true, heur, k=3))
    return ranker, {"ndcg@3_lambdarank": float(np.mean(nd_r)), "ndcg@3_heuristic_mean_saving_per_gbp": float(np.mean(nd_h)), "n_test_homes": len(nd_r)}


# ------------------------------------------------------------------ improvement potential

def fit_improvement(X, lab) -> tuple[object, dict]:
    tr, te = lab.split == "train", lab.split == "test"
    y = lab.improvement_potential_pct.values
    ok = np.isfinite(y)
    m = lgb.LGBMRegressor(**BASE_PARAMS).fit(X[tr & ok], y[tr & ok])
    pred = m.predict(X[te & ok])
    band_idx = X.loc[te & ok, "epc_band"].cat.codes.values.astype(float)
    band_idx[band_idx > 6] = np.nan  # not_sure
    known = np.isfinite(band_idx)
    rep = {
        "spearman_model": float(spearmanr(y[te & ok], pred).correlation),
        "spearman_epc_band_order_on_known": float(spearmanr(y[te & ok][known], band_idx[known]).correlation),
        "spearman_model_on_same_known_subset": float(spearmanr(y[te & ok][known], pred[known]).correlation),
        **metrics(y[te & ok], pred),
    }
    return m, rep


def main(tune: bool = False, rebuild_features: bool = False) -> dict:
    t0 = time.time()
    props, lab, X = load_dataset(rebuild_features)
    print(f"loaded {len(lab)} rows, features {X.shape}", flush=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    energy, rep_e = fit_energy(X, lab, tune)
    energy.pop("_predict_fns", None)
    joblib.dump({"models": energy, "features": list(X.columns), "created": time.strftime("%Y-%m-%d %H:%M")}, MODELS_DIR / "baseline_models.joblib")
    print("energy done", flush=True)
    feas_models, rep_f = fit_feasibility(X, lab, props)
    joblib.dump(feas_models, MODELS_DIR / "feasibility_clf.joblib")
    print("feasibility done", flush=True)
    sur, rep_s = fit_surrogates(X, lab)
    joblib.dump(sur, MODELS_DIR / "saving_surrogates.joblib")
    print("surrogates done", flush=True)
    ranker, rep_r = fit_ranker(X, lab)
    joblib.dump(ranker, MODELS_DIR / "ranker.joblib")
    ip, rep_ip = fit_improvement(X, lab)
    joblib.dump(ip, MODELS_DIR / "improvement_model.joblib")
    report = {"energy": rep_e, "feasibility": rep_f, "savings_surrogates": rep_s, "ranking": rep_r, "improvement_potential": rep_ip,
              "seconds": round(time.time() - t0, 1)}
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps({k: report["energy"][k] for k in ("electricity_test", "gas_test")}, indent=2, default=float))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="Optuna search (needs optuna)")
    ap.add_argument("--rebuild-features", action="store_true")
    a = ap.parse_args()
    main(a.tune, a.rebuild_features)
