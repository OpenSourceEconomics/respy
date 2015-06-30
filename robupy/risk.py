""" This module contains the functions related to the incorporation of
    risk in the model.
"""

# standard library
import numpy as np

# project library
from robupy._shared import _get_future_payoffs

''' Public functions
'''


def simulate_emax_risk(num_draws, period_payoffs_ex_post, eps_baseline, period,
                       k, period_payoffs_ex_ante, edu_max, edu_start,
                       mapping_state_idx, states_all, future_payoffs,
                       num_periods, emax):
    """ Simulate expected future value
    """
    # Initialize container
    simulated = 0.0

    # Calculate maximum value
    for i in range(num_draws):

        # Calculate ex post payoffs
        for j in [0, 1]:
            period_payoffs_ex_post[period, k, j] = \
                period_payoffs_ex_ante[period, k, j] * \
                np.exp(eps_baseline[period, i, j])

        for j in [2, 3]:
            period_payoffs_ex_post[period, k, j] = \
                period_payoffs_ex_ante[period, k, j] + \
                eps_baseline[period, i, j]

        # Check applicability
        if period == (num_periods - 1):
            continue

        # Get future values
        future_payoffs[period, k, :] = \
            _get_future_payoffs(edu_max, edu_start, mapping_state_idx,
                                period, emax, k, states_all)

        # Calculate total utilities
        total_payoffs = period_payoffs_ex_post[period, k, :] + \
            future_payoffs[period, k, :]

        # Determine optimal choice
        maximum = max(total_payoffs)

        # Recording expected future value
        simulated += maximum

    # Scaling
    simulated = simulated / num_draws

    # Finishing
    return simulated, period_payoffs_ex_ante, period_payoffs_ex_post, \
           future_payoffs
