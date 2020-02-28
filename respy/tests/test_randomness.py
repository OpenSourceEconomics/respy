"""This module includes test to specifically test that randomness is held constant."""
import numpy as np
import pytest

from respy.likelihood import get_crit_func
from respy.simulate import get_simulate_func
from respy.solve import get_solve_func
from respy.tests.utils import apply_to_attributes_of_two_state_spaces
from respy.tests.utils import process_model_or_seed


@pytest.mark.parametrize("model", ["kw_94_one", "kw_97_basic", "kw_2000"])
def test_invariance_of_model_solution_in_solve_and_criterion_functions(model):
    params, options = process_model_or_seed(model)

    options["n_periods"] = 2 if model == "kw_2000" else 3

    solve = get_solve_func(params, options)
    state_space = solve(params)

    simulate = get_simulate_func(params, options)
    df = simulate(params)
    state_space_sim = simulate.keywords["solve"].keywords["state_space"]

    criterion = get_crit_func(params, options, df)
    _ = criterion(params)
    state_space_crit = criterion.keywords["solve"].keywords["state_space"]

    for state_space_ in [state_space_sim, state_space_crit]:
        assert state_space.core.equals(state_space_.core)

        apply_to_attributes_of_two_state_spaces(
            state_space.get_attribute("wages"),
            state_space_.get_attribute("wages"),
            np.testing.assert_array_equal,
        )
        apply_to_attributes_of_two_state_spaces(
            state_space.get_attribute("nonpecs"),
            state_space_.get_attribute("nonpecs"),
            np.testing.assert_array_equal,
        )
        apply_to_attributes_of_two_state_spaces(
            state_space.get_attribute("expected_value_functions"),
            state_space_.get_attribute("expected_value_functions"),
            np.testing.assert_array_equal,
        )
        apply_to_attributes_of_two_state_spaces(
            state_space.get_attribute("base_draws_sol"),
            state_space_.get_attribute("base_draws_sol"),
            np.testing.assert_array_equal,
        )
