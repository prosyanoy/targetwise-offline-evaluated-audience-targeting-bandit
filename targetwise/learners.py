"""Linear contextual bandits implemented from scratch (numpy only).

Each action (0 = skip, 1 = target) has its own ridge-regression model of expected revenue.
Treating is worthwhile when the (exploration-adjusted) revenue uplift exceeds the cost.
All policies expose ``prob_treat(X, cost)`` (exact or Monte-Carlo) for offline evaluation,
``act`` for online use, ``update`` for online learning and ``fit_logged`` for learning from
logged data with inverse-propensity weights.
"""
from __future__ import annotations

import numpy as np

from .simulator import LoggedData, Simulator


def features(X: np.ndarray) -> np.ndarray:
    """Intercept, raw features and squares of the first five (mild non-linearity)."""
    return np.hstack([np.ones((len(X), 1)), X, X[:, :5] ** 2])


class LinearArm:
    """Ridge regression with sufficient statistics A = l2*I + sum w phi phi^T, b = sum w y phi."""

    def __init__(self, d: int, l2: float = 1.0):
        self.d, self.l2 = d, l2
        self.A = l2 * np.eye(d)
        self.b = np.zeros(d)
        self.Ainv = np.eye(d) / l2

    def update(self, Phi: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> None:
        w = np.ones(len(y)) if w is None else w
        self.A += (Phi * w[:, None]).T @ Phi
        self.b += Phi.T @ (w * y)
        self.Ainv = np.linalg.inv(self.A)

    @property
    def theta(self) -> np.ndarray:
        return self.Ainv @ self.b

    def width(self, Phi: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Phi, self.Ainv, Phi), 0.0))


class LinearBandit:
    name = "linear_bandit"

    def __init__(self, reward_scale: float = 9.0, l2: float = 1.0, ipw_clip: float = 10.0, seed: int = 0):
        d = features(np.zeros((1, 6))).shape[1]
        self.arms = [LinearArm(d, l2), LinearArm(d, l2)]
        self.sigma = reward_scale  # revenue noise scale, re-estimated by fit_logged
        self.ipw_clip = ipw_clip
        self.rng = np.random.default_rng(seed)

    # ---- learning ----
    def update(self, X, action, rev, weight=None) -> None:
        Phi = features(X)
        for a in (0, 1):
            m = action == a
            if m.any():
                self.arms[a].update(Phi[m], rev[m], None if weight is None else weight[m])

    def fit_logged(self, data: LoggedData) -> "LinearBandit":
        """Replay logged data with clipped inverse-propensity weights (normalised to mean 1 per arm)."""
        w = np.minimum(1.0 / data.p_taken, self.ipw_clip)
        for a in (0, 1):
            m = data.action == a
            wa = w[m] / w[m].mean()
            self.arms[a].update(features(data.X[m]), data.rev[m], wa)
        Phi = features(data.X)
        pred = np.where(data.action == 1, Phi @ self.arms[1].theta, Phi @ self.arms[0].theta)
        self.sigma = float(np.sqrt(np.average((data.rev - pred) ** 2, weights=w)))
        return self

    # ---- scoring ----
    def mean_rev(self, X, a: int) -> np.ndarray:
        return features(X) @ self.arms[a].theta

    def uplift(self, X) -> np.ndarray:
        """Posterior-mean revenue uplift (used for budget ranking)."""
        return features(X) @ (self.arms[1].theta - self.arms[0].theta)

    def budget_score(self, X) -> np.ndarray:
        """Uplift score used to rank users under a budget (each bandit applies its own exploration rule)."""
        return self.uplift(X)

    def prob_treat(self, X, cost) -> np.ndarray:
        raise NotImplementedError

    def act(self, X, cost, rng=None) -> np.ndarray:
        rng = rng or self.rng
        return (rng.random(len(X)) < self.prob_treat(X, cost)).astype(int)


class EpsilonGreedy(LinearBandit):
    name = "epsilon_greedy"

    def __init__(self, epsilon: float = 0.1, **kw):
        super().__init__(**kw)
        self.epsilon = epsilon

    def prob_treat(self, X, cost):
        greedy = (self.uplift(X) > cost).astype(float)
        return self.epsilon / 2 + (1 - self.epsilon) * greedy


class LinUCB(LinearBandit):
    name = "linucb"

    def __init__(self, alpha: float = 0.5, **kw):
        super().__init__(**kw)
        self.alpha = alpha

    def budget_score(self, X):
        """Optimistic uplift: UCB of the treat arm minus LCB of the skip arm."""
        Phi = features(X)
        bonus = self.alpha * self.sigma * (self.arms[1].width(Phi) + self.arms[0].width(Phi))
        return self.uplift(X) + bonus

    def prob_treat(self, X, cost):
        Phi = features(X)
        ucb = [Phi @ self.arms[a].theta + self.alpha * self.sigma * self.arms[a].width(Phi) for a in (0, 1)]
        return (ucb[1] - cost > ucb[0]).astype(float)


class ThompsonSampling(LinearBandit):
    name = "thompson"

    def __init__(self, n_draws: int = 64, **kw):
        super().__init__(**kw)
        self.n_draws = n_draws

    def _draw(self, a: int, n: int, rng) -> np.ndarray:
        cov = self.sigma**2 * self.arms[a].Ainv
        return rng.multivariate_normal(self.arms[a].theta, cov, size=n)

    def budget_score(self, X):
        """Uplift under one fixed posterior draw (posterior-sampling ranking)."""
        rng = np.random.default_rng(54321)
        return features(X) @ (self._draw(1, 1, rng)[0] - self._draw(0, 1, rng)[0])

    def prob_treat(self, X, cost):
        """Monte-Carlo probability that a posterior draw favours treating (fixed draws -> deterministic)."""
        rng = np.random.default_rng(12345)
        Phi = features(X)
        diff = (self._draw(1, self.n_draws, rng) - self._draw(0, self.n_draws, rng)).T
        return ((Phi @ diff) > np.asarray(cost)[:, None]).mean(axis=1)

    def act(self, X, cost, rng=None):
        """One posterior draw per call (batched Thompson sampling)."""
        rng = rng or self.rng
        t1, t0 = self._draw(1, 1, rng)[0], self._draw(0, 1, rng)[0]
        return ((features(X) @ (t1 - t0)) > cost).astype(int)


def simulate_online(policy: LinearBandit, sim: Simulator, rounds: int, batch: int, rng) -> np.ndarray:
    """Learn online from a fresh bandit stream; returns cumulative pseudo-regret at each batch end."""
    regret, cum = [], 0.0
    for _ in range(rounds // batch):
        X, cost = sim.sample_context(batch, rng)
        a = policy.act(X, cost, rng)
        policy.update(X, a, sim.observe(X, a, rng))
        best = np.maximum(sim.mu(X, 1) - cost, sim.mu(X, 0))
        got = np.where(a == 1, sim.mu(X, 1) - cost, sim.mu(X, 0))
        cum += float((best - got).sum())
        regret.append(cum)
    return np.array(regret)
