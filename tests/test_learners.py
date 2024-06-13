import numpy as np

from targetwise.learners import EpsilonGreedy, LinUCB, ThompsonSampling, features, simulate_online
from targetwise.uplift import TwoModelUplift


def _corr(sim, p, X):
    return np.corrcoef(p.uplift(X), sim.true_uplift(X))[0, 1]


def test_fit_recovers_uplift_ranking(sim, logged):
    Xt, _ = sim.sample_context(5000, np.random.default_rng(1))
    for cls in (EpsilonGreedy, LinUCB, ThompsonSampling):
        p = cls(reward_scale=9.0).fit_logged(logged)
        assert _corr(sim, p, Xt) > 0.8, cls.__name__
    assert _corr(sim, TwoModelUplift(30.0).fit(logged), Xt) > 0.7


def test_epsilon_greedy_probabilities(sim, logged):
    p = EpsilonGreedy(0.2).fit_logged(logged)
    X, c = sim.sample_context(500, np.random.default_rng(2))
    pr = p.prob_treat(X, c)
    assert set(np.round(pr, 6)) <= {0.1, 0.9}


def test_linucb_is_more_optimistic_with_larger_alpha(sim, logged):
    X, _ = sim.sample_context(500, np.random.default_rng(3))
    lo = LinUCB(0.0).fit_logged(logged).budget_score(X)
    hi = LinUCB(2.0).fit_logged(logged).budget_score(X)
    assert np.all(hi >= lo)


def test_thompson_prob_in_unit_interval_and_matches_mean_when_confident(sim, logged):
    ts = ThompsonSampling().fit_logged(logged)
    X, c = sim.sample_context(500, np.random.default_rng(4))
    pr = ts.prob_treat(X, c)
    assert pr.min() >= 0 and pr.max() <= 1
    clear = np.abs(ts.uplift(X) - c) > 3.0
    assert np.all((pr[clear] > 0.5) == (ts.uplift(X)[clear] > c[clear]))


def test_online_update_matches_batch_ridge(sim, logged):
    sub = logged.subset(np.arange(2000))
    a = EpsilonGreedy()
    a.update(sub.X, sub.action, sub.rev)
    b = EpsilonGreedy()
    for i in range(0, 2000, 500):
        b.update(sub.X[i:i + 500], sub.action[i:i + 500], sub.rev[i:i + 500])
    assert np.allclose(a.arms[1].theta, b.arms[1].theta)
    assert features(sub.X).shape[1] == a.arms[0].d


def test_online_regret_grows_sublinearly_for_learning_bandits(sim):
    r = simulate_online(ThompsonSampling(reward_scale=9.0), sim, 10_000, 500, np.random.default_rng(5))
    assert np.all(np.diff(r) >= 0)
    assert (r[-1] - r[len(r) // 2]) < r[len(r) // 2]  # later half adds less regret than first half
