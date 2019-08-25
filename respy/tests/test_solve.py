import numpy as np
import pytest
from numba import njit

import respy as rp
from respy._numba import array_to_tuple
from respy.config import EXAMPLE_MODELS
from respy.config import KEANE_WOLPIN_1994_MODELS
from respy.config import KEANE_WOLPIN_1997_MODELS
from respy.pre_processing.model_checking import check_model_solution
from respy.pre_processing.model_processing import process_params_and_options
from respy.simulate import calculate_value_functions_and_flow_utilities
from respy.solve import solve
from respy.state_space import _create_state_space
from respy.tests._former_code import _create_state_space_kw94
from respy.tests._former_code import _create_state_space_kw97_base
from respy.tests._former_code import _create_state_space_kw97_extended
from respy.tests.random_model import generate_random_model
from respy.tests.utils import process_model_or_seed


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_check_solution(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

    state_space = solve(params, options)

    params, optim_paras, options = process_params_and_options(params, options)

    check_model_solution(options, state_space)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_state_space_restrictions_by_traversing_forward(model_or_seed):
    """Test for inadmissible states in the state space.

    The test is motivated by the addition of another restriction in
    https://github.com/OpenSourceEconomics/respy/pull/145. To ensure that similar errors
    do not happen again, this test takes all states in one period and indexes each of
    the four subsequent states. If a state in the next period got hit return one else 0.
    Repeating the same procedure for all periods, we get a list of hopefully of zeros
    and ones for each state which indicates whether the state was indexed or not.

    """

    @njit
    def traverse_forward(
        states_,
        indexer_current,
        indexer_future,
        indicator_,
        is_inadmissible,
        n_choices_w_exp,
        n_lagged_choices,
    ):
        n_choices = is_inadmissible.shape[1]

        for i in range(states_.shape[0]):

            k_parent = indexer_current[array_to_tuple(indexer_current, states_[i, 1:])]

            for choice in range(n_choices):
                if is_inadmissible[k_parent, choice]:
                    pass
                else:
                    child = states_[i, 1:].copy()

                    if choice < n_choices_w_exp:
                        child[choice] += 1

                    if n_lagged_choices:
                        child[
                            n_choices_w_exp + 1 : n_choices_w_exp + n_lagged_choices
                        ] = child[
                            n_choices_w_exp : n_choices_w_exp + n_lagged_choices - 1
                        ]
                        child[n_choices_w_exp] = choice

                    k = indexer_future[array_to_tuple(indexer_future, child)]
                    indicator_[k] = 1

        return indicator_

    params, options = process_model_or_seed(model_or_seed)

    params, optim_paras, options = process_params_and_options(params, options)

    state_space = solve(params, options)

    indicator = np.zeros(state_space.states.shape[0])

    for period in range(options["n_periods"] - 1):
        states = state_space.get_attribute_from_period("states", period)

        indicator = traverse_forward(
            states,
            state_space.indexer[period],
            state_space.indexer[period + 1],
            indicator,
            state_space.is_inadmissible,
            len(options["choices_w_exp"]),
            options["n_lagged_choices"],
        )

    # Restrict indicator to states of the second period as the first is never indexed.
    n_states_first_period = state_space.get_attribute_from_period("states", 0).shape[0]
    indicator = indicator[n_states_first_period:]

    assert (indicator == 1).all()


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_invariance_of_solution(model_or_seed):
    """Test for the invariance of the solution.

    We run solve two times and check whether all attributes of the state space match.

    """
    params, options = process_model_or_seed(model_or_seed)

    params, optim_paras, options = process_params_and_options(params, options)

    state_space = solve(params, options)
    state_space_ = solve(params, options)

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
