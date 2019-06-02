from functools import partial

import numpy as np
from numba import guvectorize
from numba import vectorize

from respy.config import HUGE_FLOAT
from respy.config import INADMISSIBILITY_PENALTY
from respy.pre_processing.data_checking import check_estimation_data
from respy.pre_processing.model_processing import parse_parameters
from respy.pre_processing.model_processing import process_options
from respy.pre_processing.model_processing import process_params
from respy.shared import create_base_draws
from respy.shared import get_conditional_probabilities
from respy.solve import solve_with_backward_induction
from respy.solve import StateSpace


def get_crit_func(params, options, df):
    """Get the criterion function.

    The return value is the :func:`log_like` where all arguments except the
    parameter vector are fixed with ``functools.partial``. Thus, the function can be
    passed directly into any optimization algorithm.

    Parameters
    ----------
    params : pd.DataFrame
        DataFrame containing model parameters.
    options : dict
        Dictionary containing model options.
    df : pd.DataFrame
        The model is fit to this dataset.

    Returns
    -------
    criterion_function : func
        Criterion function where all arguments except the parameter vector are set.

    Raises
    ------
    AssertionError
        If data has not the expected format.

    """
    params, optim_paras = process_params(params)
    options = process_options(options)

    check_estimation_data(options, df)

    state_space = StateSpace(params, options)

    # Collect arguments for estimation.
    base_draws_est = create_base_draws(
        (options["num_periods"], options["estimation_draws"], 4),
        options["estimation_seed"],
    )

    criterion_function = partial(
        log_like,
        interpolation_points=options["interpolation_points"],
        data=df,
        tau=options["estimation_tau"],
        base_draws_est=base_draws_est,
        state_space=state_space,
    )
    return criterion_function


def log_like(params, interpolation_points, data, tau, base_draws_est, state_space):
    """Criterion function for the likelihood maximization.

    This function calculates the average likelihood contribution of the sample.

    Parameters
    ----------
    params : Series
        Parameter Series
    interpolation : bool
        Indicator for the interpolation routine.
    num_points_interp : int
        Number
    data : pd.DataFrame
        The log likelihood is calculated for this data.
    tau : float
        Smoothing parameter.
    base_draws_est : np.ndarray
        Set of draws to calculate the probability of observed wages.
    state_space : class
        State space.

    """
    optim_paras = parse_parameters(params)

    state_space.update_systematic_rewards(optim_paras)

    state_space = solve_with_backward_induction(
        state_space, interpolation_points, optim_paras
    )

    contribs = log_like_obs(state_space, data, base_draws_est, tau, optim_paras)

    crit_val = -np.mean(np.clip(np.log(contribs), -HUGE_FLOAT, HUGE_FLOAT))

    return crit_val


@vectorize("f8(f8, f8, f8)", nopython=True, target="cpu")
def clip(x, minimum=None, maximum=None):
    """Clip (limit) input value.

    Parameters
    ----------
    x : float
        Value to be clipped.
    minimum : float
        Lower limit.
    maximum : float
        Upper limit.

    Returns
    -------
    float
        Clipped value.

    """
    if minimum is not None and x < minimum:
        return minimum
    elif maximum is not None and x > maximum:
        return maximum
    else:
        return x


@guvectorize(
    ["f8[:], f8[:], f8[:], f8[:, :], f8, b1, i8, f8, f8[:]"],
    "(m), (n), (n), (p, n), (), (), (), () -> (p)",
    nopython=True,
    target="parallel",
)
def simulate_probability_of_agents_observed_choice(
    wages, nonpec, emaxs, draws, delta, max_education, idx, tau, prob_choice
):
    """Simulate the probability of observing the agent's choice.

    The probability is simulated by iterating over a distribution of unobservables.
    First, the utility of each choice is computed. Then, the probability of observing
    the choice of the agent given the maximum utility from all choices is computed.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (2,).
    nonpec : np.ndarray
        Array with shape (4,).
    emaxs : np.ndarray
        Array with shape (4,)
    draws : np.ndarray
        Array with shape (num_draws, 4)
    delta : float
        Discount rate.
    max_education: bool
        Indicator for whether the state has reached maximum education.
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
                rew_ex = wages[j] * draws[i, j] + nonpec[j]
            else:
                rew_ex = nonpec[j] + draws[i, j]

            cont_value = rew_ex + delta * emaxs[j]

            if j == 2 and max_education:
                cont_value += INADMISSIBILITY_PENALTY

            total_values[j, i] = cont_value

            if cont_value > max_total_values or j == 0:
                max_total_values = cont_value

        sum_smooth_values = 0.0

        for j in range(num_choices):
            val_exp = np.exp((total_values[j, i] - max_total_values) / tau)

            val_clipped = clip(val_exp, 0.0, HUGE_FLOAT)

            total_values[j, i] = val_clipped
            sum_smooth_values += val_clipped

        prob_choice[i] = total_values[idx, i] / sum_smooth_values


@vectorize(["f4(f4, f4, f4)", "f8(f8, f8, f8)"], nopython=True, target="cpu")
def normal_pdf(x, mu, sigma):
    """Compute the probability of ``x`` under a normal distribution.

    This implementation is faster than calling ``scipy.stats.norm.pdf``.

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
    >>> result = normal_pdf(0)
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
    base_draws_est,
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
    base_draws_est : np.ndarray
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
    num_draws, num_choices = base_draws_est.shape[1:]
    temp_draws = np.zeros((num_draws, num_choices))

    # Extract relevant deviates from standard normal distribution. The same set of
    # baseline draws are used for each agent and period.
    draws_stan = base_draws_est[period]

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

            prob_wages[:] = normal_pdf(dist, 0.0, sc[0, 0])

        # Adjust draws and prob_wages in case of OCCUPATION B.
        elif choice == 2:
            temp_draws[:, 0] = draws_stan[:, 0]
            temp_draws[:, 1] = (dist - sc[1, 0] * draws_stan[:, 0]) / sc[1, 1]

            means = sc[1, 0] * draws_stan[:, 0]
            prob_wages[:] = normal_pdf(dist, means, sc[1, 1])

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


def log_like_obs(state_space, data, base_draws_est, tau, optim_paras):
    """Calculate the likelihood contribution of each individual in the sample.

    The function calculates all likelihood contributions for all observations in the
    data which means all individual-period-type combinations. Then, likelihoods are
    accumulated within each individual and type over all periods. After that, the result
    is multiplied with the type-specific shares which yields the contribution to the
    likelihood for each individual.

    Parameters
    ----------
    state_space : class
        Class of state space.
    data : pd.DataFrame
        DataFrame with the empirical dataset.
    base_draws_est : np.ndarray
        Array with shape (num_periods, num_draws_est, num_choices) containing i.i.d.
        draws from standard normal distributions.
    tau : float
        Smoothing parameter for choice probabilities.
    optim_paras : dict
        Dictionary with quantities that were extracted from the parameter vector.

    Returns
    -------
    contribs : np.ndarray
        Array with shape (num_agents,) containing contributions of estimated agents.

    """
    if np.count_nonzero(optim_paras["shocks_cholesky"]) == 0:
        return np.ones(data.Identifier.unique().shape[0])

    # Convert data to np.ndarray. Separate wages from other characteristics as they need
    # to be integers.
    agents = data[
        [
            "Period",
            "Experience_A",
            "Experience_B",
            "Years_Schooling",
            "Lagged_Choice",
            "Choice",
        ]
    ].values.astype(int)
    wages_observed = data["Wage"].values

    # Get the number of observations for each individual and an array with indices of
    # each individual's first observation. After that, extract initial education levels
    # per agent which are important for type-specific probabilities.
    num_obs_per_agent = np.bincount(data.Identifier.values)
    idx_agents_first_observation = np.hstack((0, np.cumsum(num_obs_per_agent)[:-1]))
    agents_initial_education_levels = agents[idx_agents_first_observation, 3]

    # Update type-specific probabilities conditional on whether the initial level of
    # education is greater than nine.
    type_shares = get_conditional_probabilities(
        optim_paras["type_shares"], agents_initial_education_levels
    )

    # Extract observable components of the state space and agent's decision.
    periods, exp_as, exp_bs, edus, choices_lagged, choices = (
        agents[:, i] for i in range(6)
    )

    # Get indices of states in the state space corresponding to all observations for all
    # types. The indexer has the shape (num_obs, num_types).
    ks = state_space.indexer[periods, exp_as, exp_bs, edus, choices_lagged - 1, :]

    # Reshape periods, choices and wages_observed so that they match the shape (num_obs,
    # num_types) of the indexer.
    periods = state_space.states[ks, 0]
    choices = choices.repeat(state_space.num_types).reshape(-1, state_space.num_types)
    wages_observed = wages_observed.repeat(state_space.num_types).reshape(
        -1, state_space.num_types
    )
    wages_systematic = state_space.wages[ks]

    # Adjust the draws to simulate the expected maximum utility and calculate the
    # probability of observing the wage.
    draws, prob_wages = create_draws_and_prob_wages(
        wages_observed,
        wages_systematic,
        periods,
        base_draws_est,
        choices,
        optim_paras["shocks_cholesky"],
    )

    # Simulate the probability of observing the choice of the individual.
    prob_choices = simulate_probability_of_agents_observed_choice(
        state_space.wages[ks],
        state_space.nonpec[ks],
        state_space.emaxs[ks, :4],
        draws,
        optim_paras["delta"],
        state_space.states[ks, 3] >= state_space.edu_max,
        choices - 1,
        tau,
    )

    # Multiply the probability of the agent's choice with the probability of wage and
    # average over all draws to get the probability of the observation.
    prob_obs = (prob_choices * prob_wages).mean(axis=2)

    # Accumulate the likelihood of observations for each individual-type combination
    # over all periods.
    prob_type = np.multiply.reduceat(prob_obs, idx_agents_first_observation)

    # Multiply each individual-type contribution with its type-specific shares and sum
    # over types to get the likelihood contribution for each individual.
    contribs = (prob_type * type_shares).sum(axis=1)

    return contribs
