"""Auxiliary functions for the evaluation of the likelihood."""
import numpy as np

from numba import guvectorize
from respy.python.shared.shared_constants import HUGE_FLOAT


@guvectorize(
    ["float64[:], int64, float64, float64[:]"],
    "(n), (), () -> ()",
    nopython=True,
)
def get_smoothed_probability(total_values, idx, tau, prob_choice):
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

    Example
    -------
    >>> get_smoothed_probability(np.zeros(4), 3, 0.5)
    0.25

    >>> total_values = np.arange(1, 9).reshape(2, 4)
    >>> get_smoothed_probability(total_values, 3, 0.5)
    array([0.86495488, 0.86495488])

    """
    prob_choice[0] = 0.0
    num_choices = total_values.shape[0]

    max_total_values = -HUGE_FLOAT
    for i in range(num_choices):
        if total_values[i] > max_total_values:
            max_total_values = total_values[i]

    sum_smooth_values = 0.0

    for i in range(num_choices):
        temp = np.exp((total_values[i] - max_total_values) / tau)
        if temp > HUGE_FLOAT:
            temp = HUGE_FLOAT
        elif temp < 0.0:
            temp = 0.0

        total_values[i] = temp
        sum_smooth_values += temp

    prob_choice[0] = total_values[idx] / sum_smooth_values
