""" This module contains the functions related to the incorporation of
    risk in the model.
"""

# standard library
import numpy as np

# project library
from robupy.shared import *

''' Public functions
'''


def simulate_emax_risk(num_draws, payoffs_ex_post, eps_baseline, period,
        k, payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
        states_all, future_payoffs, num_periods, emax, delta):
    """ Simulate expected future value under risk.
    """
    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant = eps_baseline.copy()
    for j in [0, 1]:
        eps_relevant[:, j] = np.exp(eps_relevant[:, j])

    # Simulate the expected future value for a given parameterization.
    simulated, payoffs_ex_post, future_payoffs = simulate_emax(num_draws,
            payoffs_ex_post, period, k, eps_relevant, payoffs_ex_ante, edu_max,
            edu_start, num_periods, emax, states_all, future_payoffs,
            mapping_state_idx, delta)

    # Finishing
    return simulated, payoffs_ex_post, future_payoffs
