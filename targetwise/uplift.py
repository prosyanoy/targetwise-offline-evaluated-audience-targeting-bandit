"""Two-model (T-learner) uplift baseline: one conversion classifier per action."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .simulator import LoggedData


class TwoModelUplift:
    name = "uplift_two_model"

    def __init__(self, value: float = 30.0, seed: int = 0):
        self.value = value
        self.models = [
            HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=3, random_state=seed)
            for _ in (0, 1)
        ]

    def fit(self, data: LoggedData) -> "TwoModelUplift":
        y = (data.rev > 0).astype(int)
        for a in (0, 1):
            m = data.action == a
            self.models[a].fit(data.X[m], y[m])
        return self

    def mean_rev(self, X, a: int) -> np.ndarray:
        return self.value * self.models[a].predict_proba(X)[:, 1]

    def uplift(self, X) -> np.ndarray:
        return self.mean_rev(X, 1) - self.mean_rev(X, 0)

    def budget_score(self, X) -> np.ndarray:
        return self.uplift(X)

    def prob_treat(self, X, cost) -> np.ndarray:
        return (self.uplift(X) > cost).astype(float)

    def act(self, X, cost, rng=None) -> np.ndarray:
        return self.prob_treat(X, cost).astype(int)
