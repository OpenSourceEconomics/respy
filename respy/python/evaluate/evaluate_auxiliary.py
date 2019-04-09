"""Auxiliary functions for the evaluation of the likelihood."""
import numpy as np

from numba import guvectorize, vectorize
from respy.python.shared.shared_constants import HUGE_FLOAT, INADMISSIBILITY_PENALTY


@vectorize("f8(f8, f8, f8)", nopython=True, target="cpu")
def clip(x, min_=None, max_=None):
    """Clip (limit) input value.

    Parameters
    ----------
    x : float
        Value to be clipped.
    min_ : float
        Lower limit.
    max_ : float
        Upper limit.

    Returns
    -------
    float
        Clipped value.

    """
    if min_ is not None and x < min_:
        return min_
    elif max_ is not None and x > max_:
        return max_
    else:
        return x


@guvectorize(
    ["f8[:], f8[:], f8[:], f8[:, :], f8, b1, i8, f8, f8[:]"],
    "(m), (n), (n), (p, n), (), (), (), () -> (p)",
    nopython=True,
    target="parallel",
)
def simulate_probability_of_agents_observed_choice(
    wages, rewards_systematic, emaxs, draws, delta, max_education, idx, tau, prob_choice
):
    """Simulate the probability of observing the agent's choice.

    The probability is simulated by iterating over a distribution of unobservables.
    First, the utility of each choice is computed. Then, the probability of observing
    the choice of the agent given the maximum utility from all choices is computed.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (2,).
    rewards_systematic : np.ndarray
        Array with shape (4,).
    emaxs : np.ndarray
        Array with shape (4,)
    draws : np.ndarray
        Array with shape (num_draws, 4)
    delta : float
        Discount rate.
    idx : int
        Choice of the agent minus one to get an index.
    tau : float
        Smoothing parameter for choice probabilities.

    Returns
    -------
    prob_choice : np.ndarray
        Array with shape (num_draws) containing smoothed probabilities for
        choice.

    """
    num_draws, num_choices = draws.shape
    num_wages = wages.shape[0]

    total_values = np.zeros((num_choices, num_draws))

    for i in range(num_draws):

        max_total_values = 0.0

        for j in range(num_choices):
            if j < num_wages:
                rew_ex = (
                    wages[j] * draws[i, j] + rewards_systematic[j] - wages[j]
                )
            else:
                rew_ex = rewards_systematic[j] + draws[i, j]

            if j == 2 and max_education:
                rew_ex += INADMISSIBILITY_PENALTY

            total_values[j, i] = rew_ex + delta * emaxs[j]

            if total_values[j, i] > max_total_values or j == 0:
                max_total_values = total_values[j, i]

        sum_smooth_values = 0.0

        for j in range(num_choices):
            val_exp = np.exp((total_values[j, i] - max_total_values) / tau)

            val_clipped = clip(val_exp, 0.0, HUGE_FLOAT)

            total_values[j, i] = val_clipped
            sum_smooth_values += val_clipped

        prob_choice[i] = total_values[idx, i] / sum_smooth_values


@vectorize(["float64(float64, float64, float64)"], nopython=True, target="cpu")
def get_pdf_of_normal_distribution(x, mu, sigma):
    """Compute the probability of ``x`` under a normal distribution.

    This implementation is faster than calling :func:`scipy.stats.norm.pdf`.

    Parameters
    ----------
    x : float or np.ndarray
        The probability is calculated for this value.
    mu : float or np.ndarray
        Mean of the normal distribution.
    sigma : float or np.ndarray
        Standard deviation of the normal distribution.

    Returns
    -------
    probability : float
        Probability of ``x`` under a normal distribution with mean ``mu`` and standard
        deviation ``sigma``.

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
    wages_systematic,
    period,
    periods_draws_prob,
    choice,
    sc,
    draws,
    prob_wages,
):
    """Create draws to simulate maximum utilities and create probabilities of wages.

    Draws are taken from a general set of unique shocks for each period. The shocks are
    adjusted in case the wage of an agent is available as well as the probability of the
    wage.

    Parameters
    ----------
    wage_observed : float
        Agent's observed wage.
    wages_systematic : np.ndarray
        Array with shape (2,) containing systematic wages.
    period : int
        Number of period.
    periods_draws_prob : np.ndarray
        Array with shape (num_periods, num_draws, num_choices) containing sets of draws
        for all periods.
    choice : int
        Choice between one and four.
    sc : np.ndarray
        Array with shape (num_choices, num_choices).

    Returns
    -------
    draws : np.ndarray
        Array with shape (num_draws, num_choices) containing adjusted draws.
    prob_wages : np.ndarray
        Array with shape (num_draws,) containing probabilities for the observed wage.

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
        log_wage_observed = clip(log_wo, -HUGE_FLOAT, HUGE_FLOAT)

        log_ws = np.log(wages_systematic[choice - 1])
        log_wage_systematic = clip(log_ws, -HUGE_FLOAT, HUGE_FLOAT)

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
    # done by adjusting the loops. Additionally, it incorporates the process of taking
    # ``np.exp`` for the draws of the first two choices and clipping them.
    k_, l_ = draws.shape
    m_ = sc.shape[0]

    for k in range(k_):
        for m in range(m_):
            val = 0.0
            for l in range(l_):
                val += temp_draws[k, l] * sc[m, l]
            if m < 2:
                val_exp = np.exp(val)
                val_clipped = clip(val_exp, 0.0, HUGE_FLOAT)

                draws[k, m] = val_clipped
            else:
                draws[k, m] = val
