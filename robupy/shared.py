""" This module contains functionality that is shared across the main modules of the package.
"""


# standard library
import numpy as np

def simulate_emax(num_draws, period, k, eps_relevant, payoffs_ex_ante, edu_max,
                  edu_start, num_periods, emax, states_all, mapping_state_idx,
                  delta):
    """ Simulate expected future value.
    """

    # Initialize containers
    payoffs_ex_post, emax_simulated = np.tile(np.nan, 4), 0.0

    # Calculate maximum value
    for i in range(num_draws):

        # Calculate ex post payoffs
        for j in [0, 1]:
            payoffs_ex_post[j] = payoffs_ex_ante[j] * eps_relevant[i, j]

        for j in [2, 3]:
            payoffs_ex_post[j] = payoffs_ex_ante[j] + eps_relevant[i, j]

        # Check applicability
        if period == (num_periods - 1):
            future_payoffs = np.tile(np.nan, 4)
            continue

        # Get future values
        future_payoffs = get_future_payoffs(edu_max, edu_start,
                            mapping_state_idx, period, emax, k, states_all)

        # Calculate total utilities
        total_payoffs = payoffs_ex_post + delta * future_payoffs

        # Determine optimal choice
        maximum = max(total_payoffs)

        # Recording expected future value
        emax_simulated += maximum

    # Scaling
    emax_simulated = emax_simulated / num_draws

    # Finishing
    return emax_simulated, payoffs_ex_post, future_payoffs


def get_future_payoffs(edu_max, edu_start, mapping_state_idx, period, emax, k,
                        states_all):
    """ Get future payoffs for additional choices.
    """

    # Distribute state space
    exp_A, exp_B, edu, edu_lagged = states_all[period, k, :]

    # Future utilities
    future_payoffs = np.tile(np.nan, 4)

    # Working in occupation A
    future_idx = mapping_state_idx[period + 1, exp_A + 1, exp_B,
                                   edu, 0]
    future_payoffs[0] = emax[period + 1, future_idx]

    # Working in occupation B
    future_idx = mapping_state_idx[period + 1, exp_A, exp_B + 1,
                                   edu, 0]

    future_payoffs[1] = emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year
    # of schooling is only possible for those that have strictly
    # less than the maximum level of additional education allowed.
    if edu < edu_max - edu_start:

        future_idx = mapping_state_idx[period + 1, exp_A, exp_B,
                                       edu + 1, 1]

        future_payoffs[2] = emax[period + 1, future_idx]

    else:

        future_payoffs[2] = -np.inf

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_A, exp_B,
                                   edu, 0]

    future_payoffs[3] = emax[period + 1, future_idx]

    # Ensuring that schooling does not increase beyond the
    # maximum allowed level. This is necessary as in the
    # special case where delta is equal to zero,
    # (-np.inf * 0) evaluates to NAN.
    if edu >= edu_max - edu_start:
        future_payoffs[2] = -np.inf

    # Finishing
    return future_payoffs
