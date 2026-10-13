"""Grouped splits and leakage control (spec 6.10).

GroupShuffleSplit by customer: ~70/15/15. An out-of-distribution test set holds out one
region and one age band entirely.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

OOD_REGION = "wales"
OOD_AGE_BAND = "B"

# Never features: reference labels, hidden state, post-retrofit values, future sales.
NOT_AVAILABLE_PREFIXES = ("clean_", "hidden_", "d_", "pkg_", "pct_saving", "saving_per_gbp", "bill_saving", "improvement_potential",
                          "sap_points", "epc_band_true", "hlc", "hlp", "feasible_", "monthly_", "obs_", "state_json")


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    split = pd.Series("train", index=df.index)
    ood = (df["region"] == OOD_REGION) | (df["age_band"] == OOD_AGE_BAND)
    split[ood] = "ood_test"
    # other properties of customers who own an OOD home are excluded so no customer straddles splits
    ood_customers = set(df.loc[ood, "customer_id"])
    straddle = (~ood) & df["customer_id"].isin(ood_customers)
    split[straddle] = "excluded"
    rest = df[~ood & ~straddle]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, hold_idx = next(gss.split(rest, groups=rest["customer_id"]))
    hold = rest.iloc[hold_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    val_idx, test_idx = next(gss2.split(hold, groups=hold["customer_id"]))
    split[hold.index[val_idx]] = "val"
    split[hold.index[test_idx]] = "test"
    return split


def check_no_group_leak(df: pd.DataFrame) -> None:
    used = df[df.split != "excluded"]
    groups = used.groupby("customer_id")["split"].nunique()
    if (groups > 1).any():
        raise AssertionError(f"customers in several splits: {list(groups[groups > 1].index[:5])}")


def is_allowed_feature(col: str) -> bool:
    return not col.startswith(NOT_AVAILABLE_PREFIXES)
