""" This module contains functionality that is shared across the main modules of the package.
"""


# standard library
import numpy as np

# project library
import robupy.fort.performance as perf


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
        future_payoffs = perf.get_future_payoffs(edu_max, edu_start, mapping_state_idx, period, emax, k, states_all)

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


