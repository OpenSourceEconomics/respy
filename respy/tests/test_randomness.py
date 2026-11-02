"""This module includes test to specifically test that randomness is held constant."""

import numpy as np
import pytest

from respy.likelihood import get_log_like_func
from respy.simulate import get_simulate_func
from respy.solve import get_solve_func
from respy.tests.utils import apply_to_attributes_of_two_state_spaces
from respy.tests.utils import process_model_or_seed


@pytest.mark.end_to_end
@pytest.mark.parametrize(
    "model",
    ["robinson_crusoe_extended", "robinson_crusoe_with_observed_characteristics"],
)
def test_invariance_of_model_solution_in_solve_and_criterion_functions(model):
    params, options = process_model_or_seed(model)

    solve = get_solve_func(params, options)
    state_space = solve(params)

    simulate = get_simulate_func(params, options)
    df = simulate(params)
    state_space_sim = simulate.keywords["solve"].keywords["state_space"]

    log_like = get_log_like_func(params, options, df)
    _ = log_like(params)
    state_space_crit = log_like.keywords["solve"].keywords["state_space"]

    for state_space_ in [state_space_sim, state_space_crit]:
        assert state_space.core.equals(state_space_.core.reindex_like(state_space.core))

        apply_to_attributes_of_two_state_spaces(
            state_space.wages,
            state_space_.wages,
            np.testing.assert_array_equal,
        )
        apply_to_attributes_of_two_state_spaces(
            state_space.nonpecs,
            state_space_.nonpecs,
            np.testing.assert_array_equal,
        )
        apply_to_attributes_of_two_state_spaces(
            state_space.expected_value_functions,
            state_space_.expected_value_functions,
            np.testing.assert_array_equal,
        )
        apply_to_attributes_of_two_state_spaces(
            state_space.base_draws_sol,
            state_space_.base_draws_sol,
            np.testing.assert_array_equal,
        )
