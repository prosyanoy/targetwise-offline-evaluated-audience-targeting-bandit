import numpy as np

from targetwise import ope
from targetwise.simulator import LoggedData
from targetwise.uplift import TwoModelUplift


def _oracle_q(sim, d):
    return sim.mu(d.X, 0), sim.mu(d.X, 1)


def test_logging_policy_ope_equals_sample_mean(logged):
    e = ope.ips(logged, logged.propensity)
    assert np.isclose(e.value, logged.reward.mean())


def test_estimators_close_to_truth_with_ci_covering(sim):
    rng = np.random.default_rng(10)
    pol = lambda X, c: (sim.true_uplift(X) > c).astype(float)  # oracle deterministic policy
    truth = sim.true_policy_value(pol, np.random.default_rng(11), n=300_000)
    cover = {"IPS": 0, "SNIPS": 0, "DR": 0}
    runs = 30
    for _ in range(runs):
        d = sim.generate_logged(8000, rng)
        q0, q1 = _oracle_q(sim, d)
        for k, e in ope.evaluate_all(d, pol(d.X, d.cost), q0, q1).items():
            cover[k] += e.lo <= truth <= e.hi
    for k, v in cover.items():
        assert v / runs >= 0.8, (k, v)


def test_dr_has_lower_variance_than_ips(sim):
    d = sim.generate_logged(20_000, np.random.default_rng(12))
    pi1 = (sim.true_uplift(d.X) > d.cost).astype(float)
    q0, q1 = _oracle_q(sim, d)
    assert ope.doubly_robust(d, pi1, q0, q1).se < ope.ips(d, pi1).se


def test_dr_unbiased_with_bad_reward_model(sim):
    rng = np.random.default_rng(13)
    pol = lambda X, c: (X[:, 3] > 0).astype(float)
    truth = sim.true_policy_value(pol, np.random.default_rng(14), n=300_000)
    vals = []
    for _ in range(20):
        d = sim.generate_logged(10_000, rng)
        vals.append(ope.doubly_robust(d, pol(d.X, d.cost), np.zeros(d.n), np.zeros(d.n)).value)
    assert abs(np.mean(vals) - truth) < 0.15


def test_naive_mean_is_biased_but_ips_is_not(sim, logged):
    pi1 = np.ones(logged.n)
    truth = sim.true_policy_value(lambda X, c: np.ones(len(X)), np.random.default_rng(15))
    assert abs(ope.ips(logged, pi1).value - truth) < 0.3
    assert abs(logged.reward[logged.action == 1].mean() - truth) > abs(ope.ips(logged, pi1).value - truth)


def test_snips_bounded_by_reward_range(logged):
    e = ope.snips(logged, np.ones(logged.n), clip=5.0)
    assert logged.reward.min() <= e.value <= logged.reward.max()


def test_paired_difference_se_smaller_than_unpaired(sim, logged):
    q = TwoModelUplift(30.0).fit(logged.subset(np.arange(20_000)))
    ev = logged.subset(np.arange(20_000, 40_000))
    q0, q1 = q.mean_rev(ev.X, 0), q.mean_rev(ev.X, 1)
    a = ope.doubly_robust(ev, q.prob_treat(ev.X, ev.cost), q0, q1)
    b = ope.doubly_robust(ev, ev.propensity, q0, q1)
    diff = ope.difference(a, b)
    assert np.isclose(diff.value, a.value - b.value)
    assert diff.lo <= diff.value <= diff.hi
    assert diff.se < np.hypot(a.se, b.se)
