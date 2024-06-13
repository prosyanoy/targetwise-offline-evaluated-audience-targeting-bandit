import numpy as np

from targetwise.budget import BudgetedTargeter, select_budgeted


def test_selects_best_ratio_within_budget():
    uplift = np.array([4.0, 3.0, 1.0, 0.5, 10.0])
    cost = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    m = select_budgeted(uplift, cost, budget=2.0)
    assert list(m) == [True, True, False, False, False]  # ratios 4,3 beat 2 (the big one)
    assert cost[m].sum() <= 2.0


def test_never_targets_below_min_roi():
    m = select_budgeted(np.array([0.5, 0.9]), np.array([1.0, 1.0]), budget=100.0)
    assert not m.any()


def test_spend_never_exceeds_budget_and_monotone():
    rng = np.random.default_rng(0)
    u, c = rng.normal(2, 2, 1000), rng.uniform(0.5, 2, 1000)
    prev = -1
    for b in (10, 50, 200, 5000):
        m = select_budgeted(u, c, b)
        assert c[m].sum() <= b
        assert m.sum() >= prev
        prev = m.sum()


def test_targeter_calibration_respects_budget(sim, logged):
    from targetwise.learners import LinUCB
    s = LinUCB().fit_logged(logged)
    X, c = sim.sample_context(20_000, np.random.default_rng(1))
    t = BudgetedTargeter(s, 0.2).calibrate(X, c)
    assert t.spend_per_user(X, c) <= 0.2 + 1e-9
    X2, c2 = sim.sample_context(20_000, np.random.default_rng(2))
    assert t.spend_per_user(X2, c2) <= 0.2 * 1.1
    assert t.spend_per_user(X, c) > 0.15  # budget is actually used
