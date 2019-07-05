import numpy as np
import pytest
from numba import njit

from respy._numba import array_to_tuple
from respy.config import EXAMPLE_MODELS
from respy.config import KEANE_WOLPIN_1994_MODELS
from respy.config import KEANE_WOLPIN_1997_MODELS
from respy.pre_processing.model_checking import check_model_solution
from respy.pre_processing.model_processing import process_params_and_options
from respy.solve import get_continuation_values
from respy.solve import solve
from respy.solve import StateSpace
from respy.state_space import _create_state_space
from respy.tests._former_code import _create_state_space_kw94
from respy.tests._former_code import _create_state_space_kw97
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
    def traverse_forward(states_, indexer, indicator_, is_inadmissible):
        n_choices_w_exp = states_.shape[1] - 3
        n_choices = is_inadmissible.shape[1]

        for i in range(states_.shape[0]):

            k_parent = indexer[array_to_tuple(indexer, states_[i])]

            for n in range(n_choices):
                if is_inadmissible[k_parent, n]:
                    pass
                else:
                    child = states_[i].copy()
                    # Change to future period.
                    child[0] += 1

                    if n < n_choices_w_exp:
                        child[n + 1] += 1

                    child[-2] = n

                    k = indexer[array_to_tuple(indexer, child)]
                    indicator_[k] = 1

        return indicator_

    params, options = process_model_or_seed(model_or_seed)

    params, optim_paras, options = process_params_and_options(params, options)

    state_space = solve(params, options)

    indicator = np.zeros(state_space.states.shape[0])

    for period in range(options["n_periods"] - 1):
        states = state_space.get_attribute_from_period("states", period)

        indicator = traverse_forward(
            states, state_space.indexer, indicator, state_space.is_inadmissible
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
        state_space.continuation_values, state_space_.continuation_values
    )
    np.testing.assert_array_equal(
        state_space.base_draws_sol, state_space_.base_draws_sol
    )


@pytest.mark.parametrize("model_or_seed", range(10))
def test_get_continuation_values(model_or_seed):
    """Test propagation of emaxs from last to first period."""
    params, options = process_model_or_seed(model_or_seed)

    params, optim_paras, options = process_params_and_options(params, options)

    state_space = StateSpace(params, options)

    n_states_last_period = state_space.get_attribute_from_period(
        "states", options["n_periods"] - 1
    ).shape[0]
    n_states_but_last_period = state_space.states.shape[0] - n_states_last_period

    state_space.continuation_values = np.r_[
        np.zeros((n_states_but_last_period, len(options["choices"]))),
        np.ones((n_states_last_period, len(options["choices"]))),
    ]
    state_space.emax_value_functions = np.r_[
        np.zeros(n_states_but_last_period), np.ones(n_states_last_period)
    ]

    for period in reversed(range(options["n_periods"] - 1)):
        states = state_space.get_attribute_from_period("states", period)
        state_space.continuation_values = get_continuation_values(
            states,
            state_space.indexer,
            state_space.continuation_values,
            state_space.emax_value_functions,
            state_space.is_inadmissible,
        )
        state_space.emax_value_functions = state_space.continuation_values.max(axis=1)

    assert (state_space.emax_value_functions == 1).all()
    assert state_space.continuation_values.mean() >= 0.95


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
    states_new.lagged_choice = states_new.lagged_choice.replace(
        {choice: i for i, choice in enumerate(options["choices"])}
    )

    # Compare the state spaces via sets as ordering changed in some cases.
    states_old_set = set(map(tuple, states_old))
    states_new_set = set(map(tuple, states_new.to_numpy()))
    assert states_old_set == states_new_set

    # Compare indexers via masks for valid indices.
    mask_old = indexer_old != -1
    mask_new = indexer_new != -1
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
    states_old, indexer_old = _create_state_space_kw97(
        n_periods, n_types, edu_starts, edu_max
    )

    states_new, indexer_new = _create_state_space(options)
    states_new.lagged_choice = states_new.lagged_choice.replace(
        {choice: i for i, choice in enumerate(options["choices"])}
    )

    # Compare the state spaces via sets as ordering changed in some cases.
    states_old_set = set(map(tuple, states_old))
    states_new_set = set(map(tuple, states_new.to_numpy()))
    assert states_old_set == states_new_set

    # Compare indexers via masks for valid indices.
    mask_old = indexer_old != -1
    mask_new = indexer_new != -1
    assert np.array_equal(mask_old, mask_new)
