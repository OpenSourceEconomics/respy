""" This module contains the functions related to the incorporation of
    risk in the model.
"""

# standard library
import numpy as np

# project library
from robupy.checks.checks_risk import checks_risk

import robupy.performance.access as perf

''' Public functions
'''


def simulate_emax_risk(num_draws, eps_baseline, period,
        k, payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
        states_all, num_periods, emax, delta, fast, debug,
                       ambiguity_args=None):
    """ Simulate expected future value under risk.
    """
    # Check input parameters
    if debug is True:
        checks_risk('simulate_emax_risk', ambiguity_args)

    # Access performance library
    perf_lib = perf.get_library(fast)

    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant = eps_baseline.copy()
    for j in [0, 1]:
        eps_relevant[:, j] = np.exp(eps_relevant[:, j])

    # Simulate expected future value.
    simulated, payoffs_ex_post, future_payoffs = perf_lib.simulate_emax(num_periods, num_draws, period, k, eps_relevant,
                                 payoffs_ex_ante, edu_max, edu_start, emax, states_all,
                                 mapping_state_idx, delta)

    # Finishing
    return simulated, payoffs_ex_post, future_payoffs
