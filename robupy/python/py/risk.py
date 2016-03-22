""" This module contains the functions related to the incorporation of
    risk in the model.
"""
# standard library
import numpy as np

# project library
from robupy.python.py.auxiliary import simulate_emax


''' Public functions
'''


def get_payoffs_risk(num_draws_emax, draws_emax, period, k,
                     payoffs_systematic, edu_max, edu_start, mapping_state_idx,
                     states_all, num_periods, periods_emax, delta, shocks_cholesky):
    """ Simulate expected future value under risk.
    """
    # Auxiliary object
    shocks_mean = np.tile(0.0, 2)
    # Simulate expected future value.
    emax, payoffs_ex_post, payoffs_future = simulate_emax(num_periods,
                                                          num_draws_emax, period, k, draws_emax, payoffs_systematic,
                                                          edu_max, edu_start, periods_emax, states_all, mapping_state_idx,
                                                          delta, shocks_cholesky, shocks_mean)

    # Finishing
    return emax, payoffs_ex_post, payoffs_future
