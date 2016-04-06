""" This module contains some auxiliary functions for the evaluation of the
criterion function.
"""

# standard library
import numpy as np

# project library
from robupy.python.shared.shared_auxiliary import check_dataset
from robupy.python.shared.shared_constants import HUGE_FLOAT


''' Auxiliary functions
'''


def get_smoothed_probability(total_payoffs, idx, tau):
    """ Construct the smoothed choice probabilities.
    """
    maxim_payoff = max(total_payoffs)

    smoot_payoff = np.clip(np.exp((total_payoffs - maxim_payoff)/tau), 0.0,
        HUGE_FLOAT)

    prob_choice = smoot_payoff[idx] / sum(smoot_payoff)

    # Finishing
    return prob_choice


def check_output(crit_val):
    """ Check integrity of criterion function.
    """
    assert (np.isfinite(crit_val))
    assert (isinstance(crit_val, float))

    # Finishing
    return True


def check_input(robupy_obj, data_frame):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert robupy_obj.get_attr('is_locked')

    if robupy_obj.get_attr('is_solved'):
        robupy_obj.reset()

    # Check that dataset aligns with model specification.
    check_dataset(data_frame, robupy_obj)

    # Finishing
    return True
