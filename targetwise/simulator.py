"""Simulated logged-data generator with a known logging policy and heterogeneous effects.

Actions: 0 = do not target, 1 = target (costs ``cost`` per targeted user).
Outcome: conversion y ~ Bernoulli(p_a(x)); revenue = value * y.
Net reward = revenue - cost * action.
The logging policy is confounded (depends on x0, x4) and its propensities are logged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

DIM = 6


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LoggedData:
    X: np.ndarray
    cost: np.ndarray
    action: np.ndarray  # 0/1
    propensity: np.ndarray  # logging P(action=1 | x)
    rev: np.ndarray

    @property
    def n(self) -> int:
        return len(self.action)

    @property
    def p_taken(self) -> np.ndarray:
        return np.where(self.action == 1, self.propensity, 1.0 - self.propensity)

    @property
    def reward(self) -> np.ndarray:
        return self.rev - self.cost * self.action

    def subset(self, idx) -> "LoggedData":
        return LoggedData(self.X[idx], self.cost[idx], self.action[idx], self.propensity[idx], self.rev[idx])

    def split(self, frac: float, rng: np.random.Generator) -> tuple["LoggedData", "LoggedData"]:
        perm = rng.permutation(self.n)
        k = int(self.n * frac)
        return self.subset(perm[:k]), self.subset(perm[k:])


class Simulator:
    def __init__(self, value: float = 30.0):
        self.value = value

    # ---- ground truth (never shown to learners) ----
    def p_conv(self, X: np.ndarray, a: int) -> np.ndarray:
        base = -2.0 + 0.6 * X[:, 0] + 0.4 * X[:, 1] - 0.3 * X[:, 2] + 0.2 * X[:, 0] * X[:, 1]
        if a == 0:
            return sigmoid(base)
        g = 0.7 * (0.9 * X[:, 3] + 0.6 * X[:, 1] - 0.5 * np.maximum(X[:, 2], 0) - 0.2)
        return sigmoid(base + g)

    def mu(self, X: np.ndarray, a: int) -> np.ndarray:
        """Expected revenue under action a."""
        return self.value * self.p_conv(X, a)

    def true_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.mu(X, 1) - self.mu(X, 0)

    # ---- generation ----
    def sample_context(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        X = rng.normal(size=(n, DIM))
        cost = 0.5 + 1.5 * sigmoid(X[:, 4])
        return X, cost

    def logging_propensity(self, X: np.ndarray) -> np.ndarray:
        return np.clip(sigmoid(0.2 + 0.5 * X[:, 0] - 0.4 * X[:, 4]), 0.1, 0.9)

    def observe(self, X: np.ndarray, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = np.where(action == 1, self.p_conv(X, 1), self.p_conv(X, 0))
        return self.value * (rng.random(len(action)) < p)

    def generate_logged(self, n: int, rng: np.random.Generator) -> LoggedData:
        X, cost = self.sample_context(n, rng)
        prop = self.logging_propensity(X)
        action = (rng.random(n) < prop).astype(int)
        return LoggedData(X, cost, action, prop, self.observe(X, action, rng))

    # ---- oracle evaluation ----
    def expected_reward(self, X, cost, pi1) -> np.ndarray:
        """Per-user expected net reward of a (stochastic) policy with P(treat)=pi1."""
        return pi1 * (self.mu(X, 1) - cost) + (1 - pi1) * self.mu(X, 0)

    def true_policy_value(self, prob_treat: Callable, rng: np.random.Generator, n: int = 200_000) -> float:
        X, cost = self.sample_context(n, rng)
        return float(self.expected_reward(X, cost, prob_treat(X, cost)).mean())


class LoggingPolicy:
    name = "logging_policy"

    def __init__(self, sim: Simulator):
        self.sim = sim

    def prob_treat(self, X, cost):
        return self.sim.logging_propensity(X)


class FixedPolicy:
    def __init__(self, p: float, name: str):
        self.p, self.name = p, name

    def prob_treat(self, X, cost):
        return np.full(len(X), self.p)
