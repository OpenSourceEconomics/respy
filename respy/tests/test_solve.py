import numpy as np
import pytest
from numba import njit

from respy.config import EXAMPLE_MODELS
from respy.pre_processing.model_checking import check_model_solution
from respy.pre_processing.model_processing import process_model_spec
from respy.shared import get_example_model
from respy.solve import get_emaxs_of_subsequent_period
from respy.solve import solve
from respy.solve import StateSpace
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_check_solution(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    state_space = solve(params_spec, options_spec)

    attr, optim_paras = process_model_spec(params_spec, options_spec)

    check_model_solution(attr, optim_paras, state_space)


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
    def traverse_forward(states, indexer, indicator, edu_max):
        for i in range(states.shape[0]):
            # Unpack parent state and get index.
            period, exp_a, exp_b, edu, choice_lagged, type_ = states[i]

            # Working in Occupation A in period + 1
            k = indexer[period + 1, exp_a + 1, exp_b, edu, 0, type_]
            indicator[k] = 1

            # Working in Occupation B in period +1
            k = indexer[period + 1, exp_a, exp_b + 1, edu, 1, type_]
            indicator[k] = 1

            # Schooling in period + 1. Note that adding an additional year of schooling
            # is only possible for those that have strictly less than the maximum level
            # of additional education allowed. This condition is necessary as there are
            # states which have reached maximum education. Incrementing education by one
            # would target an inadmissible state.
            if edu >= edu_max:
                pass
            else:
                k = indexer[period + 1, exp_a, exp_b, edu + 1, 2, type_]
                indicator[k] = 1

            # Staying at home in period + 1
            k = indexer[period + 1, exp_a, exp_b, edu, 3, type_]
            indicator[k] = 1

        return indicator

    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    state_space = solve(params_spec, options_spec)

    indicator = np.zeros(state_space.num_states)

    for period in range(state_space.num_periods - 1):
        states = state_space.get_attribute_from_period("states", period)

        indicator = traverse_forward(
            states, state_space.indexer, indicator, state_space.edu_max
        )

    # Restrict indicator to states of the second period as the first is never indexed.
    indicator = indicator[state_space.states_per_period[0] :]

    assert (indicator == 1).all()


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_invariance_of_solution(model_or_seed):
    """Test for the invariance of the solution.

    We run solve two times and check whether all attributes of the state space match.

    """
    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    state_space = solve(params_spec, options_spec)
    state_space_ = solve(params_spec, options_spec)

    np.array_equal(state_space.states, state_space_.states)
    np.array_equal(state_space.covariates, state_space_.covariates)
    np.array_equal(state_space.rewards, state_space_.rewards)
    np.array_equal(state_space.emaxs, state_space_.emaxs)
    np.array_equal(state_space.base_draws_sol, state_space_.base_draws_sol)


@pytest.mark.parametrize("seed", range(10))
def test_get_emaxs_of_subsequent_period(seed):
    """Test propagation of emaxs from last to first period."""
    params_spec, options_spec = generate_random_model()

    state_space = StateSpace(params_spec, options_spec)

    state_space.emaxs = np.r_[
        np.zeros((state_space.states_per_period[:-1].sum(), 5)),
        np.full((state_space.states_per_period[-1], 5), 10),
    ]
    for period in reversed(range(state_space.num_periods - 1)):
        states = state_space.get_attribute_from_period("states", period)
        state_space.emaxs = get_emaxs_of_subsequent_period(
            states, state_space.indexer, state_space.emaxs, state_space.edu_max
        )
        state_space.emaxs[:, 4] = state_space.emaxs[:, :4].max()
    assert (state_space.emaxs == 10).all()
