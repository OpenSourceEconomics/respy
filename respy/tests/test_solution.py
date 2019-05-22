"""Test the solution routine."""
import numpy as np
import pytest
from numba import njit

from respy.python.solve.solve_auxiliary import StateSpace


@pytest.mark.parametrize(
    "num_periods, num_types, edu_starts, edu_max",
    [(15, 1, [10], 20), (15, 5, [10, 15], 20)],
)
def test_state_space_restrictions_by_traversing_forward(
    num_periods, num_types, edu_starts, edu_max
):
    """Test for inadmissible states in the state space.

    The test is motivated by the addition of another restriction in
    https://github.com/OpenSourceEconomics/respy/pull/145. To ensure that similar errors
    do not happen again, this test takes all states in one period and indexes each of
    the four subsequent states. If a state in the next period got hit return one else 0.
    Repeating the same procedure for all periods, we get a list of hopefully of zeros
    and ones for each state which indicates whether the state was indexed or not.

    """

    @njit
    def traverse_forward(states, indexer, indicator):
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

    state_space = StateSpace(num_periods, num_types, edu_starts, edu_max)

    indicator = np.zeros(state_space.num_states)

    for period in range(num_periods - 1):
        states = state_space.get_attribute_from_period("states", period)

        indicator = traverse_forward(states, state_space.indexer, indicator)

    # Restrict indicator to states of the second period as the first is never indexed.
    indicator = indicator[state_space.states_per_period[0] :]

    assert (indicator == 1).all()
