""" This module contains the simulation of the expected future value.
"""

# project library
from robupy.python.shared.shared_auxiliary import transform_disturbances
from robupy.python.shared.shared_auxiliary import get_total_value

''' Main function
'''


def simulate_emax(num_periods, num_draws_emax, period, k, draws_emax,
        payoffs_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, delta, shocks_cholesky, shocks_mean):
    """ Simulate expected future value.
    """
    # Get the transformed set of disturbances
    draws_emax_transformed = transform_disturbances(draws_emax,
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


