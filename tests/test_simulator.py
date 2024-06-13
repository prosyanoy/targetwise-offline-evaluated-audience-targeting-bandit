import numpy as np


def test_propensities_valid_and_logged(logged):
    assert logged.propensity.min() >= 0.1 and logged.propensity.max() <= 0.9
    assert abs(logged.action.mean() - logged.propensity.mean()) < 0.01
    assert np.all(logged.p_taken > 0)


def test_heterogeneous_effects_with_negative_segment(sim, logged):
    tau = sim.true_uplift(logged.X)
    assert tau.min() < 0 < tau.max()


def test_logging_is_confounded(sim, logged):
    naive = logged.rev[logged.action == 1].mean() - logged.rev[logged.action == 0].mean()
    true = sim.true_uplift(logged.X).mean()
    assert naive - true > 0.3


def test_reward_includes_cost(logged):
    assert np.allclose(logged.reward, logged.rev - logged.cost * logged.action)
