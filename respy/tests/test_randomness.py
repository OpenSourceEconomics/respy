"""This module includes test to specifically test that randomness is held constant."""
import numpy as np
import pytest

import respy as rp
from respy.config import EXAMPLE_MODELS
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(5)))
def test_invariance_of_model_solution_in_solve_and_simulation(model_or_seed):
    if isinstance(model_or_seed, str):
        params, options = rp.get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params, options = generate_random_model()

    state_space = rp.solve(params, options)
    state_space_, _ = rp.simulate(params, options)

    np.testing.assert_array_equal(state_space.states, state_space_.states)
    np.testing.assert_array_equal(state_space.wages, state_space_.wages)
    np.testing.assert_array_equal(state_space.nonpec, state_space_.nonpec)

    np.testing.assert_array_equal(state_space.emaxs, state_space_.emaxs)
    np.testing.assert_array_equal(
        state_space.base_draws_sol, state_space_.base_draws_sol
    )


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(5)))
def test_invariance_of_model_solution_in_solve_and_crit_func(model_or_seed):
    if isinstance(model_or_seed, str):
        params, options = rp.get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params, options = generate_random_model()

    state_space = rp.solve(params, options)

    _, df = rp.simulate(params, options)

    crit_func = rp.get_crit_func(params, options, df)
    crit_func(params)

    state_space_ = crit_func.keywords["state_space"]

    np.testing.assert_array_equal(state_space.states, state_space_.states)
    np.testing.assert_array_equal(state_space.wages, state_space_.wages)
    np.testing.assert_array_equal(state_space.nonpec, state_space_.nonpec)

    np.testing.assert_array_equal(state_space.emaxs, state_space_.emaxs)
    np.testing.assert_array_equal(
        state_space.base_draws_sol, state_space_.base_draws_sol
    )
