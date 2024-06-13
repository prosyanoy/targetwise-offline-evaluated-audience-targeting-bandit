import numpy as np

from targetwise import abtest
from targetwise.decision import decide
from targetwise.simulator import FixedPolicy, LoggingPolicy


def test_sample_size_matches_textbook():
    # d=0.5 sd, alpha .05 two-sided, power .8 -> ~63 per arm
    assert abs(abtest.sample_size_per_arm(0.5, 1.0) - 63) <= 1


def test_power_mde_roundtrip():
    n = abtest.sample_size_per_arm(0.3, 2.0)
    assert abs(abtest.achieved_power(0.3, 2.0, n) - 0.8) < 0.01
    assert abs(abtest.minimum_detectable_effect(n, 2.0) - 0.3) < 0.01


def test_ab_detects_real_lift_and_not_null(sim):
    rng = np.random.default_rng(0)
    good = FixedPolicy(0.0, "none")
    bad = FixedPolicy(1.0, "all")
    r = abtest.run_ab_test(sim, good, bad, 20_000, rng)  # not targeting beats targeting everyone
    assert r.lift > 0 and r.p_value < 0.01
    null = abtest.run_ab_test(sim, good, FixedPolicy(0.0, "none2"), 20_000, rng)
    assert null.p_value > 0.001 and abs(null.lift) < 4 * null.se


def test_empirical_power_close_to_analytic(sim):
    rng = np.random.default_rng(1)
    t, c = FixedPolicy(0.0, "none"), LoggingPolicy(sim)
    big = abtest.run_ab_test(sim, t, c, 100_000, rng)
    n = 3000
    analytic = abtest.achieved_power(big.lift, big.sd, n, 0.1, two_sided=False)
    emp = abtest.empirical_power(sim, t, c, n, 150, rng, alpha=0.1)
    assert abs(emp - analytic) < 0.15


def _ab(lift, se):
    return abtest.ABResult(1000, 1, 1, lift, se, (lift - 2 * se, lift + 2 * se), lift - 1.64 * se,
                           lift + 1.64 * se, 0.01, 0.3, 0.6, 5.0)


def _est(v, se):
    from targetwise.ope import Estimate
    return Estimate("x", v, se, v - 2 * se, v + 2 * se)


def test_decision_logic():
    pos = {k: _est(1.0, 0.1) for k in ("IPS", "SNIPS", "DR")}
    assert decide(pos, _ab(1.0, 0.1), 1000, 900, 0.3, 0.4).verdict == "GO"
    assert decide(pos, _ab(1.0, 0.1), 1000, 900, 0.6, 0.4).verdict == "NO-GO"  # over budget
    assert decide(pos, _ab(-1.0, 0.1), 1000, 900, 0.3, 0.4).verdict == "NO-GO"
    assert decide(pos, _ab(0.1, 0.3), 1000, 900, 0.3, 0.4).verdict == "INCONCLUSIVE"
    neg = {k: _est(-1.0, 0.1) for k in pos}
    assert decide(neg, _ab(1.0, 0.1), 1000, 900, 0.3, 0.4).verdict != "GO"
