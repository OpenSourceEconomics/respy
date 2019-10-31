"""This module includes test to specifically test that randomness is held constant."""
import numpy as np
import pytest

import respy as rp
from respy.config import EXAMPLE_MODELS
from respy.tests.utils import process_model_or_seed


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(3)))
def test_invariance_of_model_solution_in_solve_and_criterion_functions(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

    if isinstance(model_or_seed, str):
        options["n_periods"] = 5

    state_space = rp.solve(params, options)

    # Simulate data.
    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)

    criterion_functions = [
        simulate,
        rp.get_crit_func(params, options, df),
        rp.get_smm_func(params, options, np.zeros(1), lambda x: 0),
    ]

    for crit_func in criterion_functions:
        _ = crit_func(params)
        if "state_space" in crit_func.keywords:
            state_space_ = crit_func.keywords["state_space"]
        else:
            state_space_ = crit_func.keywords["simulate"].keywords["state_space"]

        np.testing.assert_array_equal(state_space.states, state_space_.states)
        np.testing.assert_array_equal(state_space.wages, state_space_.wages)
        np.testing.assert_array_equal(state_space.nonpec, state_space_.nonpec)
        np.testing.assert_array_equal(
            state_space.emax_value_functions, state_space_.emax_value_functions
        )
        np.testing.assert_array_equal(
            state_space.base_draws_sol, state_space_.base_draws_sol
        )
