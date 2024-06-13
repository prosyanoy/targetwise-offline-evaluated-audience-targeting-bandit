"""Simulated A/B test: power analysis, randomised experiment on fresh users, significance test."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .simulator import Simulator


def sample_size_per_arm(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8,
                        two_sided: bool = True) -> int:
    """Users per arm for a two-sample z-test on means (equal arms, common sd)."""
    if effect <= 0:
        raise ValueError("effect must be positive")
    za = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    zb = stats.norm.ppf(power)
    return int(np.ceil(2 * ((za + zb) * sd / effect) ** 2))


def minimum_detectable_effect(n: int, sd: float, alpha: float = 0.05, power: float = 0.8,
                              two_sided: bool = True) -> float:
    za = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    return float((za + stats.norm.ppf(power)) * sd * np.sqrt(2.0 / n))


def achieved_power(effect: float, sd: float, n: int, alpha: float = 0.05, two_sided: bool = True) -> float:
    za = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    return float(stats.norm.cdf(abs(effect) / (sd * np.sqrt(2.0 / n)) - za))


@dataclass
class ABResult:
    n_per_arm: int
    mean_treat: float
    mean_control: float
    lift: float
    se: float
    ci_two_sided: tuple[float, float]
    lower_one_sided: float
    upper_one_sided: float
    p_value: float  # one-sided, H1: treatment > control
    spend_treat: float
    spend_control: float
    sd: float


def _arm(sim: Simulator, policy, n: int, rng):
    X, cost = sim.sample_context(n, rng)
    a = (rng.random(n) < policy.prob_treat(X, cost)).astype(int)
    reward = sim.observe(X, a, rng) - cost * a
    return reward, float((cost * a).mean())


def run_ab_test(sim: Simulator, treatment, control, n_per_arm: int, rng: np.random.Generator,
                alpha: float = 0.05) -> ABResult:
    rt, st_ = _arm(sim, treatment, n_per_arm, rng)
    rc, sc = _arm(sim, control, n_per_arm, rng)
    lift = float(rt.mean() - rc.mean())
    se = float(np.sqrt(rt.var(ddof=1) / n_per_arm + rc.var(ddof=1) / n_per_arm))
    z2, z1 = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(1 - alpha)
    return ABResult(
        n_per_arm, float(rt.mean()), float(rc.mean()), lift, se,
        (lift - z2 * se, lift + z2 * se), lift - z1 * se, lift + z1 * se,
        float(stats.norm.sf(lift / se)), st_, sc, float(np.sqrt((rt.var(ddof=1) + rc.var(ddof=1)) / 2)),
    )


def empirical_power(sim, treatment, control, n_per_arm: int, n_sims: int, rng, alpha: float = 0.05) -> float:
    """Share of repeated simulated experiments that reject H0 (one-sided) -- checks the analytic power."""
    hits = sum(run_ab_test(sim, treatment, control, n_per_arm, rng, alpha).p_value < alpha for _ in range(n_sims))
    return hits / n_sims
