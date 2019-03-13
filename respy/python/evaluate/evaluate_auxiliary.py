"""Auxiliary functions for the evaluation of the likelihood."""
import numpy as np

from respy.python.shared.shared_constants import HUGE_FLOAT


def get_smoothed_probability(total_values, idx, tau):
    """Construct smoothed choice probabilities.

    Parameters
    ----------
    total_values : np.ndarray
        Array with shape (num_draws, 4).
    idx : int
        It is the choice of the agent (1-4) minus one.
    tau : float
        Smoothing parameter for choice probabilities.

    Returns
    -------
    prob_choices : np.ndarray
        Array with shape (num_draws,) containing smoothed probabilities for choice.

    """
    max_total = total_values.max(axis=1, keepdims=True)

    smoot_values = np.clip(np.exp((total_values - max_total) / tau), 0.0, HUGE_FLOAT)

    prob_choices = smoot_values[:, idx] / smoot_values.sum(axis=1)

    return prob_choices
