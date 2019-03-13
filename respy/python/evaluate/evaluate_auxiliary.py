"""Auxiliary functions for the evaluation of the likelihood."""
import numpy as np

from respy.python.shared.shared_constants import HUGE_FLOAT


def get_smoothed_probability(total_values, idx, tau):
    """Construct the smoothed choice probabilities."""
    max_total = total_values.max(axis=1, keepdims=True)

    smoot_values = np.clip(np.exp((total_values - max_total) / tau), 0.0, HUGE_FLOAT)

    prob_choices = smoot_values[:, idx] / smoot_values.sum(axis=1)

    return prob_choices
