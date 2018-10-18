"""Auxiliary functions for the evaluation of the likelihood."""
import numpy as np

from respy.python.shared.shared_constants import HUGE_FLOAT


def get_smoothed_probability(total_values, idx, tau):
    """Construct the smoothed choice probabilities."""
    maxim_values = max(total_values)

    smoot_values = np.clip(np.exp((total_values - maxim_values) / tau), 0.0, HUGE_FLOAT)

    prob_choice = smoot_values[idx] / sum(smoot_values)

    return prob_choice
