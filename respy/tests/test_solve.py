import numpy as np
import pytest

import respy as rp
from respy.config import EXAMPLE_MODELS
from respy.config import KEANE_WOLPIN_1994_MODELS
from respy.config import KEANE_WOLPIN_1997_MODELS
from respy.pre_processing.model_checking import check_model_solution
from respy.pre_processing.model_processing import process_params_and_options
from respy.simulate import calculate_value_functions_and_flow_utilities
from respy.state_space import _create_state_space
from respy.state_space import _insert_indices_of_child_states
from respy.tests._former_code import _create_state_space_kw94
from respy.tests._former_code import _create_state_space_kw97_base
from respy.tests._former_code import _create_state_space_kw97_extended
from respy.tests.random_model import generate_random_model
from respy.tests.utils import process_model_or_seed


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_check_solution(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

    state_space = rp.solve(params, options)

    params, optim_paras, options = process_params_and_options(params, options)

    check_model_solution(options, state_space)


@pytest.mark.wip
@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_state_space_restrictions_by_traversing_forward(model_or_seed):
    """Test for inadmissible states in the state space.

    The test is motivated by the addition of another restriction in
    https://github.com/OpenSourceEconomics/respy/pull/145. To ensure that similar errors
    do not happen again, this test takes all states of the first period and finds all
    their child states. Taking only the child states their children are found and so on.
    At last, the set of visited states is compared against the total set of states.

    """
    params, options = process_model_or_seed(model_or_seed)
    params, optim_paras, options = process_params_and_options(params, options)

    state_space = rp.solve(params, options)

    indices = np.full((state_space.states.shape[0], len(options["choices"])), -1)

    for period in range(options["n_periods"] - 1):

        if period == 0:
            states = state_space.get_attribute_from_period("states", period)
        else:
            indices_period = state_space.get_attribute_from_period(
                "indices_of_child_states", period - 1
            )
            indices_period = indices_period[indices_period >= 0]
            states = state_space.states[indices_period]

        indices = _insert_indices_of_child_states(
            indices,
            states,
            state_space.indexer[period],
            state_space.indexer[period + 1],
            state_space.is_inadmissible,
            len(options["choices_w_exp"]),
            options["n_lagged_choices"],
        )

    # Take all valid indices and add the indices of the first period.
    set_valid_indices = set(indices[indices >= 0]) | set(
        range(state_space.get_attribute_from_period("states", 0).shape[0])
    )

    assert set_valid_indices == set(range(state_space.states.shape[0]))


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_invariance_of_solution(model_or_seed):
    """Test for the invariance of the solution.

    We run solve two times and check whether all attributes of the state space match.

    """
    params, options = process_model_or_seed(model_or_seed)

    params, optim_paras, options = process_params_and_options(params, options)

    state_space = rp.solve(params, options)
    state_space_ = rp.solve(params, options)

    np.testing.assert_array_equal(state_space.states, state_space_.states)
    np.testing.assert_array_equal(state_space.wages, state_space_.wages)
    np.testing.assert_array_equal(state_space.nonpec, state_space_.nonpec)
    np.testing.assert_array_equal(
        state_space.emax_value_functions, state_space_.emax_value_functions
    )
    np.testing.assert_array_equal(
        state_space.base_draws_sol, state_space_.base_draws_sol
    )


@pytest.mark.parametrize("model_or_seed", KEANE_WOLPIN_1994_MODELS + list(range(10)))
def test_create_state_space_vs_specialized_kw94(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

    params, optim_paras, options = process_params_and_options(params, options)

    # Create old state space arguments.
    n_periods = options["n_periods"]
    n_types = options["n_types"]
    edu_max = options["choices"]["edu"]["max"]
    edu_starts = options["choices"]["edu"]["start"]

    # Get states and indexer from old state space.
    states_old, indexer_old = _create_state_space_kw94(
        n_periods, n_types, edu_starts, edu_max
    )

    states_new, indexer_new = _create_state_space(options)

    for i in range(1, options["n_lagged_choices"] + 1):
        states_new[f"lagged_choice_{i}"] = states_new[f"lagged_choice_{i}"].replace(
            {choice: i for i, choice in enumerate(options["choices"])}
        )

    # Compare the state spaces via sets as ordering changed in some cases.
    states_old_set = set(map(tuple, states_old))
    states_new_set = set(map(tuple, states_new.to_numpy()))
    assert states_old_set == states_new_set

    # Compare indexers via masks for valid indices.
    for period in range(n_periods):
        mask_old = indexer_old[period] != -1
        mask_new = indexer_new[period] != -1
        assert np.array_equal(mask_old, mask_new)


@pytest.mark.parametrize("model_or_seed", KEANE_WOLPIN_1997_MODELS)
def test_create_state_space_vs_specialized_kw97(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

    # Reduce runtime
    options["n_periods"] = 10 if options["n_periods"] > 10 else options["n_periods"]

    params, optim_paras, options = process_params_and_options(params, options)

    # Create old state space arguments.
    n_periods = options["n_periods"]
    n_types = options["n_types"]
    edu_max = options["choices"]["edu"]["max"]
    edu_starts = options["choices"]["edu"]["start"]

    # Get states and indexer from old state space.
    if model_or_seed == "kw_97_base":
        states_old, indexer_old = _create_state_space_kw97_base(
            n_periods, n_types, edu_starts, edu_max
        )
    else:
        states_old, indexer_old = _create_state_space_kw97_extended(
            n_periods, n_types, edu_starts, edu_max
        )

    states_new, indexer_new = _create_state_space(options)

    for i in range(1, options["n_lagged_choices"] + 1):
        states_new[f"lagged_choice_{i}"] = states_new[f"lagged_choice_{i}"].replace(
            {choice: i for i, choice in enumerate(options["choices"])}
        )

    # Compare the state spaces via sets as ordering changed in some cases.
    states_old_set = set(map(tuple, states_old))
    states_new_set = set(map(tuple, states_new.to_numpy()))
    assert states_old_set == states_new_set

    # Compare indexers via masks for valid indices.
    for period in range(n_periods):
        mask_old = indexer_old[period] != -1
        mask_new = indexer_new[period] != -1
        assert np.array_equal(mask_old, mask_new)


@pytest.mark.parametrize("seed", range(10))
def test_equality_of_total_values_and_rewexpost_for_myopic_individuals(seed):
    """Test equality of ex-post rewards and total values for myopic individuals."""
    np.random.seed(seed)

    # We need to simulate the model to get the emaxs and model attributes.
    params, options = generate_random_model(myopic=True)
    params, optim_paras, options = process_params_and_options(params, options)

    draws = np.random.randn(1, 4)

    state_space = rp.solve(params, options)

    for period in range(options["n_periods"]):
        wages = state_space.get_attribute_from_period("wages", period)
        nonpec = state_space.get_attribute_from_period("nonpec", period)
        continuation_values = state_space.get_continuation_values(period)
        is_inadmissible = state_space.get_attribute_from_period(
            "is_inadmissible", period
        )

        value_functions, flow_utilities = calculate_value_functions_and_flow_utilities(
            wages,
            nonpec,
            continuation_values,
            draws,
            optim_paras["delta"],
            is_inadmissible,
        )

        np.testing.assert_equal(
            value_functions[~is_inadmissible], flow_utilities[~is_inadmissible]
        )
