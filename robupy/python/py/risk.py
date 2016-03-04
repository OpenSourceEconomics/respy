""" This module contains the functions related to the incorporation of
    risk in the model.
"""

# project library
from robupy.python.py.auxiliary import simulate_emax


''' Public functions
'''


def get_payoffs_risk(num_draws, eps_relevant, period, k, payoffs_systematic,
        edu_max, edu_start, mapping_state_idx, states_all, num_periods,
        periods_emax, delta, is_debug, shocks, level, measure):
    """ Simulate expected future value under risk. Part of the unused
    arguments (shocks, level, measure) is  present to align the interface
    between the PYTHON and FORTRAN implementations. The other part of the
    unused arguments (is_debug) aligns the interface between risk and
    ambiguity code.
    """
    # Renaming for optimization setup, alignment with ROBUFORT
    eps_relevant_emax = eps_relevant

    # Simulate expected future value.
    simulated, payoffs_ex_post, future_payoffs = simulate_emax(num_periods,
        num_draws, period, k, eps_relevant_emax, payoffs_systematic, edu_max,
        edu_start, periods_emax, states_all, mapping_state_idx, delta)

    # Finishing
    return simulated, payoffs_ex_post, future_payoffs
