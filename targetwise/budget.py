"""Budget-constrained targeting: rank users by expected incremental value per unit cost."""
from __future__ import annotations

import numpy as np


def select_budgeted(uplift: np.ndarray, cost: np.ndarray, budget: float, min_roi: float = 1.0) -> np.ndarray:
    """Greedy knapsack: best uplift/cost first until the budget is spent.

    Only users whose expected return ratio is at least ``min_roi`` are eligible, so the
    budget is never spent on users predicted to lose money. Returns a boolean mask.
    """
    ratio = uplift / cost
    order = np.argsort(-ratio, kind="stable")
    order = order[ratio[order] >= min_roi]
    within = np.cumsum(cost[order]) <= budget
    mask = np.zeros(len(uplift), dtype=bool)
    mask[order[within]] = True
    return mask


class BudgetedTargeter:
    """Turns a scorer (anything with ``budget_score(X)``) into a per-user policy with a spend limit.

    ``calibrate`` finds the uplift/cost ratio threshold that exhausts ``budget_per_user`` on a
    reference population (contexts only, no outcomes needed); ``prob_treat`` then applies that
    threshold to any user, which makes the policy a deterministic function we can evaluate offline.
    """

    def __init__(self, scorer, budget_per_user: float, min_roi: float = 1.0):
        self.scorer, self.budget_per_user, self.min_roi = scorer, budget_per_user, min_roi
        self.threshold = np.inf
        self.name = f"{scorer.name}+budget"

    def calibrate(self, X: np.ndarray, cost: np.ndarray) -> "BudgetedTargeter":
        uplift = self.scorer.budget_score(X)
        mask = select_budgeted(uplift, cost, self.budget_per_user * len(X), self.min_roi)
        ratio = uplift / cost
        self.threshold = float(ratio[mask].min()) if mask.any() else np.inf
        return self

    def prob_treat(self, X, cost) -> np.ndarray:
        return (self.scorer.budget_score(X) / cost >= self.threshold).astype(float)

    def act(self, X, cost, rng=None) -> np.ndarray:
        return self.prob_treat(X, cost).astype(int)

    def spend_per_user(self, X, cost) -> float:
        return float((self.prob_treat(X, cost) * cost).mean())
