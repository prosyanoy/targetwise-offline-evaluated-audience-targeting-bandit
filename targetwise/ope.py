"""Off-policy evaluation: IPS, self-normalised IPS and doubly robust, with normal-approx CIs.

Target policy is described by pi1 = P(treat | x) for every logged row. Each estimate keeps its
per-sample influence terms (``psi``, mean = value) so paired differences between two policies on
the same log get a correct (covariance-aware) standard error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .simulator import LoggedData


@dataclass
class Estimate:
    name: str
    value: float
    se: float
    lo: float
    hi: float
    ess: float = float("nan")
    psi: np.ndarray = field(default=None, repr=False)


def _make(name, psi, alpha, ess=float("nan"), value=None) -> Estimate:
    n = len(psi)
    v = float(psi.mean()) if value is None else value
    se = float(psi.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1 - alpha / 2)
    return Estimate(name, v, se, v - z * se, v + z * se, ess, psi)


def importance_weights(data: LoggedData, pi1: np.ndarray, clip: float | None = None) -> np.ndarray:
    pi_taken = np.where(data.action == 1, pi1, 1.0 - pi1)
    w = pi_taken / data.p_taken
    return w if clip is None else np.minimum(w, clip)


def _ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum()) if (w**2).sum() > 0 else 0.0


def ips(data: LoggedData, pi1, alpha=0.05, clip=None) -> Estimate:
    w = importance_weights(data, pi1, clip)
    return _make("IPS", w * data.reward, alpha, _ess(w))


def snips(data: LoggedData, pi1, alpha=0.05, clip=None) -> Estimate:
    w = importance_weights(data, pi1, clip)
    r = data.reward
    v = float((w * r).sum() / w.sum())
    # linearisation: influence of the ratio estimator
    psi = v + w * (r - v) / w.mean()
    return _make("SNIPS", psi, alpha, _ess(w), value=v)


def doubly_robust(data: LoggedData, pi1, q0, q1, alpha=0.05, clip=None) -> Estimate:
    """q0, q1: model-predicted expected *revenue* under action 0 / 1 (cost is known exactly)."""
    Q0, Q1 = q0, q1 - data.cost
    w = importance_weights(data, pi1, clip)
    q_taken = np.where(data.action == 1, Q1, Q0)
    dm = pi1 * Q1 + (1 - pi1) * Q0
    return _make("DR", dm + w * (data.reward - q_taken), alpha, _ess(w))


def evaluate_all(data, pi1, q0, q1, alpha=0.05, clip=None) -> dict[str, Estimate]:
    return {
        "IPS": ips(data, pi1, alpha, clip),
        "SNIPS": snips(data, pi1, alpha, clip),
        "DR": doubly_robust(data, pi1, q0, q1, alpha, clip),
    }


def difference(a: Estimate, b: Estimate, alpha=0.05, name: str | None = None) -> Estimate:
    """Paired difference a - b computed on the same log (same estimator family)."""
    est = _make(name or f"{a.name} diff", a.psi - b.psi, alpha)
    est.ess = min(a.ess, b.ess)
    return est
