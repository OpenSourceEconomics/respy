"""Auxiliary functions for the evaluation of the likelihood."""
import numpy as np

from numba import guvectorize, njit
from respy.python.shared.shared_constants import HUGE_FLOAT


@guvectorize(
    ["float64[:, :, :], int64, float64, float64[:, :]"],
    "(m, p, n), (), () -> (m, p)",
    nopython=True,
    target="parallel",
)
def get_smoothed_probability(total_values, idx, tau, prob_choice):
    """Construct smoothed choice probabilities.

    Parameters
    ----------
    total_values : np.ndarray
        Array with shape (num_types, num_draws, num_choices).
    idx : int
        It is the choice of the agent minus one to get an index.
    tau : float
        Smoothing parameter for choice probabilities.

    Returns
    -------
    prob_choices : np.ndarray
        Array with shape (num_types, num_draws) containing smoothed probabilities for
        choice.

    """
    num_types, num_draws, num_choices = total_values.shape

    for t in range(num_types):

        for i in range(num_draws):

            max_total_values = 0.0
            for j in range(num_choices):
                if total_values[t, i, j] > max_total_values or j == 0:
                    max_total_values = total_values[t, i, j]

            sum_smooth_values = 0.0

            for j in range(num_choices):
                temp = np.exp((total_values[t, i, j] - max_total_values) / tau)
                if temp > HUGE_FLOAT:
                    temp = HUGE_FLOAT
                elif temp < 0.0:
                    temp = 0.0

                total_values[t, i, j] = temp
                sum_smooth_values += temp

            prob_choice[t, i] = total_values[t, i, idx] / sum_smooth_values


@njit(nogil=True)
def get_pdf_of_normal_distribution(x, mu=0, sigma=1):
    """Return the probability of :data:`x` assuming the normal distribution.

    This implementation is faster than calling :func:`scipy.stats.norm.pdf`.

    Parameters
    ----------
    x : float or np.ndarray
        The probability is calculated for this value.
    mu : float or np.ndarray
        Mean of the normal distribution.
    sigma : float or np.ndarray
        Standard deviation of the normal distribution.

    Example
    -------
    >>> result = get_pdf_of_normal_distribution(0)
    >>> result
    0.3989422804014327
    >>> from scipy.stats import norm
    >>> assert result == norm.pdf(0)

    """
    return (
        1
        / (np.sqrt(2 * np.pi) * sigma)
        * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    )
