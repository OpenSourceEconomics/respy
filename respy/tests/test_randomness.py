"""This module includes test to specifically test that randomness is held constant."""
import numpy as np
import pytest

import respy as rp
from respy.config import EXAMPLE_MODELS
from respy.tests.utils import process_model_or_seed


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(5)))
def test_invariance_of_model_solution_in_solve_and_simulation(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

    state_space = rp.solve(params, options)
    simulate = rp.get_simulate_func(params, options)
    _ = simulate(params)
    state_space_ = simulate.keywords["state_space"]

    np.testing.assert_array_equal(state_space.states, state_space_.states)
    np.testing.assert_array_equal(state_space.wages, state_space_.wages)
    np.testing.assert_array_equal(state_space.nonpec, state_space_.nonpec)
    np.testing.assert_array_equal(
        state_space.emax_value_functions, state_space_.emax_value_functions
    )
    np.testing.assert_array_equal(
        state_space.base_draws_sol, state_space_.base_draws_sol
    )


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(5)))
def test_invariance_of_model_solution_in_solve_and_crit_func(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

    state_space = rp.solve(params, options)

    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)

    state_space = simulate.keywords["state_space"]

    crit_func = rp.get_crit_func(params, options, df)
    crit_func(params)

    state_space_ = crit_func.keywords["state_space"]

    np.testing.assert_array_equal(state_space.states, state_space_.states)
    np.testing.assert_array_equal(state_space.wages, state_space_.wages)
    np.testing.assert_array_equal(state_space.nonpec, state_space_.nonpec)
    np.testing.assert_array_equal(
        state_space.emax_value_functions, state_space_.emax_value_functions
    )
    np.testing.assert_array_equal(
        state_space.base_draws_sol, state_space_.base_draws_sol
    )
