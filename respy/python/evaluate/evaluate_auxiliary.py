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
    ["f8, f8[:], i8, f8[:, :, :], i8, f8[:, :], f8[:, :], f8[:]"],
    "(), (w), (), (i, p, n), (), (m, n) -> (p, n), (p)",
    nopython=True,
    target="parallel",
)
def create_draws_and_prob_wages(
    wage_observed,
    wage_systematic,
    period,
    periods_draws_prob,
    choice,
    sc,
    draws,
    prob_wages,
):
    """Create draws to simulate continuation values and probabilities of wages.

    Draws are taken from a general set of unique shocks for each period. The shocks are
    adjusted in case the wage of an agent is available as well as the probability of the
    wage.

    """
    # Create auxiliary objects
    num_draws, num_choices = periods_draws_prob.shape[1:]
    temp_draws = np.zeros((num_draws, num_choices))

    # Extract relevant deviates from standard normal distribution. The same set of
    # baseline draws are used for each agent and period.
    draws_stan = periods_draws_prob[period]

    has_wage = ~np.isnan(wage_observed)

    # If an agent is observed working, then the the labor market shocks are observed and
    # the conditional distribution is used to determine the choice probabilities if the
    # wage information is available as well.
    if has_wage:
        log_wo = np.log(wage_observed)
        if log_wo > HUGE_FLOAT:
            log_wage_observed = HUGE_FLOAT
        elif log_wo < -HUGE_FLOAT:
            log_wage_observed = -HUGE_FLOAT
        else:
            log_wage_observed = log_wo

        log_ws = np.log(wage_systematic[choice - 1])
        if log_ws > HUGE_FLOAT:
            log_wage_systematic = HUGE_FLOAT
        elif log_ws < -HUGE_FLOAT:
            log_wage_systematic = -HUGE_FLOAT
        else:
            log_wage_systematic = log_ws

        dist = log_wage_observed - log_wage_systematic

        # Adjust draws and prob_wages in case of OCCUPATION A.
        if choice == 1:
            temp_draws[:, 0] = dist / sc[0, 0]
            temp_draws[:, 1] = draws_stan[:, 1]

            prob_wages[:] = get_pdf_of_normal_distribution(dist, 0.0, sc[0, 0])

        # Adjust draws and prob_wages in case of OCCUPATION B.
        elif choice == 2:
            temp_draws[:, 0] = draws_stan[:, 0]
            temp_draws[:, 1] = (dist - sc[1, 0] * draws_stan[:, 0]) / sc[1, 1]

            means = sc[1, 0] * draws_stan[:, 0]
            prob_wages[:] = get_pdf_of_normal_distribution(
                dist, means, sc[1, 1]
            )

        temp_draws[:, 2:] = draws_stan[:, 2:]

    # If the wage is missing or an agent is pursuing SCHOOLING or HOME, the draws are
    # not adjusted and the probability for wages is one.
    else:
        temp_draws[:, :] = draws_stan
        prob_wages[:] = 1.0

    # What follows is a matrix multiplication written out of the form ``a.dot(b.T). Note
    # that the second argument corresponds to ``sc`` which is not transposed. This is
    # done by adjusting the loops. The function achieves major speed performance due to
    # the check for a diagonal matrix ``b``. Additionally, it incorporates the process
    # of taking ``np.exp`` for the draws of the first two choices and clipping them.

    # Check if matrix ``b`` is diagonal.
    diag_sum = 0.0
    mat_sum = 0.0
    for i in range(sc.shape[0]):
        for j in range(sc.shape[0]):
            mat_sum += sc[i, j]
            if i == j:
                diag_sum += sc[i, j]

    diagonal = diag_sum == mat_sum

    if diagonal:
        k_ = num_draws
        m_ = sc.shape[0]

        for k in range(k_):
            for m in range(m_):
                val = temp_draws[k, m] * sc[m, m]
                if m < 2:
                    val_exp = np.exp(val)
                    if val_exp < 0:
                        val_exp = 0
                    elif val_exp > HUGE_FLOAT:
                        val_exp = HUGE_FLOAT
                    draws[k, m] = val_exp
                else:
                    draws[k, m] = val

    else:
        k_, l_ = draws.shape
        m_ = sc.shape[0]

        for k in range(k_):
            for m in range(m_):
                val = 0.0
                for l in range(l_):
                    val += temp_draws[k, l] * sc[m, l]
                if m < 2:
                    val_exp = np.exp(val)
                    if val_exp < 0:
                        val_exp = 0
                    elif val_exp > HUGE_FLOAT:
                        val_exp = HUGE_FLOAT

                    draws[k, m] = val_exp
                else:
                    draws[k, m] = val
