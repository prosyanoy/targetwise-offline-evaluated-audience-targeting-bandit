import numpy as np
import pytest

from targetwise.simulator import Simulator


@pytest.fixture(scope="module")
def sim():
    return Simulator(30.0)


@pytest.fixture(scope="module")
def logged(sim):
    return sim.generate_logged(40_000, np.random.default_rng(0))
