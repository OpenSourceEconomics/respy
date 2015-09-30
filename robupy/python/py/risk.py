""" This module contains the functions related to the incorporation of
    risk in the model.
"""

# project library
from robupy.python.py.auxiliary import simulate_emax


''' Public functions
'''


def get_payoffs_risk(num_draws, eps_relevant, period, k, payoffs_ex_ante,
        edu_max, edu_start, mapping_state_idx, states_all, num_periods, emax,
        delta):
    """ Simulate expected future value under risk.
    """
    # Renaming for optimization setup, alignment with ROBUFORT
    eps_relevant_emax = eps_relevant

    # Simulate expected future value.
    simulated, payoffs_ex_post, future_payoffs = simulate_emax(num_periods,
        num_draws, period, k, eps_relevant_emax, payoffs_ex_ante, edu_max,
        edu_start, emax, states_all, mapping_state_idx, delta)

    # Finishing
    return simulated, payoffs_ex_post, future_payoffs
