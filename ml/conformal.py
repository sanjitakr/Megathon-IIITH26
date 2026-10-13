"""Conformalized quantile regression (split-conformal CQR, Romano et al. 2019).

Fit lower/upper LightGBM quantile models on train, compute conformity scores
E = max(q_lo - y, y - q_hi) on a held-out calibration split, and widen both bounds by the
(1 - alpha)(1 + 1/n) empirical quantile of E. Gives marginal coverage >= 1 - alpha.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np


class CQR:
    def __init__(self, alpha: float = 0.2, params: dict | None = None):
        self.alpha = alpha
        self.params = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 30, "verbose": -1, **(params or {})}
        self.lo = lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2, **self.params)
        self.hi = lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha / 2, **self.params)
        self.qhat = 0.0

    def fit(self, X, y):
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        return self

    def calibrate(self, X_cal, y_cal):
        lo, hi = self.lo.predict(X_cal), self.hi.predict(X_cal)
        scores = np.maximum(lo - y_cal, y_cal - hi)
        n = len(scores)
        level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        self.qhat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.lo.predict(X) - self.qhat, self.hi.predict(X) + self.qhat
        return np.minimum(lo, hi), np.maximum(lo, hi)


def coverage(y, lo, hi) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))
