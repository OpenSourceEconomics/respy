""" This module contains the simulation of the expected future value.
"""
# standard library
import numpy as np

# project library
from robupy.shared.auxiliary import get_total_value
from robupy.shared.constants import HUGE_FLOAT

''' Main function
'''


def simulate_emax(num_periods, num_draws_emax, period, k, draws_emax,
        payoffs_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, delta, shocks_cholesky, shocks_mean):
    """ Simulate expected future value.
    """
    # Get the transformed set of disturbances
    draws_emax_transformed = _transform_disturbances(draws_emax,
        shocks_cholesky, shocks_mean)

    # Calculate maximum value
    emax_simulated = 0.0
    for i in range(num_draws_emax):

        # Select draws for this draw
        draws = draws_emax_transformed[i, :]

        # Get total value of admissible states
        total_payoffs = get_total_value(period, num_periods, delta,
            payoffs_systematic, draws, edu_max, edu_start, mapping_state_idx,
            periods_emax, k, states_all)

        # Determine optimal choice
        maximum = max(total_payoffs)

        # Recording expected future value
        emax_simulated += maximum

    # Scaling
    emax_simulated = emax_simulated / num_draws_emax

    # Finishing
    return emax_simulated



def _transform_disturbances(draws_emax, shocks_cholesky, shocks_mean):
    """ Transform the standard normal deviates to the relevant distribution.
    """
    # Transfer draws to relevant distribution
    draws_emax_transformed = draws_emax.copy()
    draws_emax_transformed = np.dot(shocks_cholesky, draws_emax_transformed.T).T
    draws_emax_transformed[:, :2] = draws_emax_transformed[:, :2] + shocks_mean

    for j in [0, 1]:
        draws_emax_transformed[:, j] = \
            np.clip(np.exp(draws_emax_transformed[:, j]), 0.0, HUGE_FLOAT)

    # Finishing
    return draws_emax_transformed