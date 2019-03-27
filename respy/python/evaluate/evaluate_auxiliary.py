"""Auxiliary functions for the evaluation of the likelihood."""
import numpy as np

from numba import guvectorize, vectorize
from respy.python.shared.shared_constants import HUGE_FLOAT


@guvectorize(
    ["float64[:, :], int64, float64, float64[:]"],
    "(p, n), (), () -> (p)",
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
    num_draws, num_choices = total_values.shape

    for i in range(num_draws):

        max_total_values = 0.0
        for j in range(num_choices):
            if total_values[i, j] > max_total_values or j == 0:
                max_total_values = total_values[i, j]

        sum_smooth_values = 0.0

        for j in range(num_choices):
            temp = np.exp((total_values[i, j] - max_total_values) / tau)
            if temp > HUGE_FLOAT:
                temp = HUGE_FLOAT
            elif temp < 0.0:
                temp = 0.0

            total_values[i, j] = temp
            sum_smooth_values += temp

        prob_choice[i] = total_values[i, idx] / sum_smooth_values


@vectorize(["float64(float64, float64, float64)"], nopython=True, target="cpu")
def get_pdf_of_normal_distribution(x, mu, sigma):
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
    a = np.sqrt(2 * np.pi) * sigma
    b = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    probability = 1 / a * b

    return probability


@guvectorize(
    ["i8, f8[:, :, :], i8, f8, f8[:, :], f8[:, :], f8[:]"],
    "(), (i, p, n), (), (), (m, n) -> (p, n), (p)",
    nopython=True,
    target="parallel",
)
def adjust_draws_and_create_prob_wages(
    period, periods_draws_prob, choice, dist, sc, draws, prob_wages
):
    # Extract relevant deviates from standard normal distribution. The same set of
    # baseline draws are used for each agent and period.
    draws_stan = periods_draws_prob[period]

    has_wage = ~np.isnan(dist)

    # If an agent is observed working, then the the labor market shocks are observed and
    # the conditional distribution is used to determine the choice probabilities if the
    # wage information is available as well.
    if has_wage:
        # Adjust draws and prob_wages in case of OCCUPATION A.
        if choice == 1:
            draws[:, 0] = dist / sc[0, 0]
            draws[:, 1] = draws_stan[:, 1]

            prob_wages[:] = get_pdf_of_normal_distribution(dist, 0.0, sc[0, 0])

        # Adjust draws and prob_wages in case of OCCUPATION B.
        elif choice == 2:
            draws[:, 0] = draws_stan[:, 0]
            draws[:, 1] = (dist - sc[1, 0] * draws_stan[:, 0]) / sc[1, 1]

            means = sc[1, 0] * draws_stan[:, 0]
            prob_wages[:] = get_pdf_of_normal_distribution(
                dist, means, sc[1, 1]
            )

        draws[:, 2:] = draws_stan[:, 2:]

    # If the wage is missing or an agent is pursuing SCHOOLING or HOME, the draws are
    # not adjusted and the probability for wages is one.
    else:
        draws[:, :] = draws_stan
        prob_wages[:] = 1.0


@guvectorize(
    ["float64[:, :], float64[:, :], float64[:, :]"],
    "(k, l), (l, m) -> (k, m)",
    nopython=True,
    target="parallel",
)
def create_draws_for_monte_carlo_simulation(a, b, out):
    """Create draws .

    This function exploits the fact that the second matrix can be a diagonal matrix
    which cuts runtime roughly by half. In the other case, it is as fast as
    ``np.tensordot``. The clip is also faster than ``np.clip`` and can be extracted as a
    ``@vectorize`` function. Combining both operations is even faster.

    This implementation can be replaced with::

        draws = np.tensordot(draws_stan, sc, axes=(3, 0))
        draws[: ,:, :, :2] = np.clip(draws[:, :, :, :2])

    """
    diag_sum = 0.0
    mat_sum = 0.0
    for i in range(b.shape[0]):
        for j in range(b.shape[0]):
            mat_sum += b[i, j]
            if i == j:
                diag_sum += b[i, j]

    diagonal = diag_sum == mat_sum

    if diagonal:
        k_ = a.shape[0]
        m_ = b.shape[1]

        for k in range(k_):
            for m in range(m_):
                val = a[k, m] * b[m, m]
                if m < 2:
                    val_exp = np.exp(val)
                    if val_exp < 0:
                        val_exp = 0
                    elif val_exp > HUGE_FLOAT:
                        val_exp = HUGE_FLOAT
                    out[k, m] = val_exp
                else:
                    out[k, m] = val

    else:
        k_, l_ = a.shape
        m_ = b.shape[1]

        for k in range(k_):
            for m in range(m_):
                temp = 0.0
                for l in range(l_):
                    temp += a[k, l] * b[l, m]
                if m < 2:
                    val_exp = np.exp(temp)
                    if val_exp < 0:
                        val_exp = 0
                    elif val_exp > HUGE_FLOAT:
                        val_exp = HUGE_FLOAT

                    out[k, m] = val_exp
                else:
                    out[k, m] = temp
