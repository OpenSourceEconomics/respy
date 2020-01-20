"""This module includes test to specifically test that randomness is held constant."""
import numpy as np
import pytest

import respy as rp
from respy.config import EXAMPLE_MODELS
from respy.tests.utils import process_model_or_seed


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(3)))
def test_invariance_of_model_solution_in_solve_and_criterion_functions(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

    options["n_periods"] = 2 if model_or_seed == "kw_2000" else 3

    state_space = rp.solve(params, options)

    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)
    state_space_sim = simulate.keywords["state_space"]

    criterion = rp.get_crit_func(params, options, df)
    _ = criterion(params)
    state_space_crit = criterion.keywords["state_space"]

    for state_space_ in [state_space_sim, state_space_crit]:
        assert state_space.core.equals(state_space_.core)
        assert state_space.dense == state_space_.dense
        # TODO
        # np.testing.assert_array_equal(state_space.wages, state_space_.wages)
        # np.testing.assert_array_equal(state_space.nonpecs, state_space_.nonpecs)
        # np.testing.assert_array_equal(
        #     state_space.expected_value_functions, state_space_.expected_value_functions
        # )
        # np.testing.assert_array_equal(
        #     state_space.base_draws_sol, state_space_.base_draws_sol
        # )
