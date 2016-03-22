""" This module contains some auxiliary functions for the PYTHON
implementations of the core functions.
"""

# standard library
import numpy as np

# project library
from robupy.constants import HUGE_FLOAT


''' Main functions
'''


def simulate_emax(num_periods, num_draws_emax, period, k, draws_emax,
        payoffs_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, delta, shocks_cholesky, shocks_mean):
    """ Simulate expected future value.
    """
    # Initialize containers
    emax_simulated, payoffs_ex_post, payoffs_future = 0.0, 0.0, 0.0

    # Transfer draws to relevant distribution
    draws_emax_transformed = draws_emax.copy()
    draws_emax_transformed = \
        np.dot(shocks_cholesky, draws_emax_transformed.T).T
    draws_emax_transformed[:, :2] = \
        draws_emax_transformed[:, :2] + shocks_mean
    for j in [0, 1]:
        draws_emax_transformed[:, j] = \
            np.clip(np.exp(draws_emax_transformed[:, j]), 0.0, HUGE_FLOAT)

    # Calculate maximum value
    for i in range(num_draws_emax):

        # Select draws for this draw
        draws = draws_emax_transformed[i, :]

        # Get total value of admissible states
        total_payoffs, payoffs_ex_post, payoffs_future = get_total_value(period,
            num_periods, delta, payoffs_systematic, draws, edu_max,
            edu_start, mapping_state_idx, periods_emax, k, states_all)

        # Determine optimal choice
        maximum = max(total_payoffs)

        # Recording expected future value
        emax_simulated += maximum

    # Scaling
    emax_simulated = emax_simulated / num_draws_emax

    # Finishing
    return emax_simulated, payoffs_ex_post, payoffs_future


def get_total_value(period, num_periods, delta, payoffs_systematic, draws,
        edu_max, edu_start, mapping_state_idx, periods_emax, k, states_all):
    """ Get total value of all possible states.
    """
    # Initialize containers
    payoffs_ex_post = np.tile(np.nan, 4)

    # Calculate ex post payoffs
    for j in [0, 1]:
        payoffs_ex_post[j] = payoffs_systematic[j] * draws[j]

    for j in [2, 3]:
        payoffs_ex_post[j] = payoffs_systematic[j] + draws[j]

    # Get future values
    if period != (num_periods - 1):
        payoffs_future = _get_future_payoffs(edu_max, edu_start,
            mapping_state_idx, period, periods_emax, k, states_all)
    else:
        payoffs_future = np.tile(0.0, 4)

    # Calculate total utilities
    total_payoffs = payoffs_ex_post + delta * payoffs_future

    # This is required to ensure that the agent does not choose any
    # inadmissible states.
    if payoffs_future[2] == -HUGE_FLOAT:
        total_payoffs[2] = -HUGE_FLOAT

    # Finishing
    return total_payoffs, payoffs_ex_post, payoffs_future


''' Auxiliary functions
'''


def _get_future_payoffs(edu_max, edu_start, mapping_state_idx, period,
        periods_emax, k, states_all):
    """ Get future payoffs for additional choices.
    """
    # Distribute state space
    exp_a, exp_b, edu, edu_lagged = states_all[period, k, :]

    # Future utilities
    payoffs_future = np.tile(np.nan, 4)

    # Working in occupation A
    future_idx = mapping_state_idx[period + 1, exp_a + 1, exp_b, edu, 0]
    payoffs_future[0] = periods_emax[period + 1, future_idx]

    # Working in occupation B
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b + 1, edu, 0]
    payoffs_future[1] = periods_emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year
    # of schooling is only possible for those that have strictly
    # less than the maximum level of additional education allowed.
    if edu < edu_max - edu_start:
        future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu + 1, 1]
        payoffs_future[2] = periods_emax[period + 1, future_idx]
    else:
        payoffs_future[2] = -HUGE_FLOAT

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu, 0]
    payoffs_future[3] = periods_emax[period + 1, future_idx]

    # Finishing
    return payoffs_future
