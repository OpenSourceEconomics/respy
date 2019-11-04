import warnings
from functools import partial

import numba as nb
import numpy as np
from scipy.special import logsumexp
from scipy.special import softmax

from respy.conditional_draws import create_draws_and_log_prob_wages
from respy.config import HUGE_FLOAT
from respy.pre_processing.data_checking import check_estimation_data
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import aggregate_keane_wolpin_utility
from respy.shared import create_base_draws
from respy.shared import create_type_covariates
from respy.shared import generate_column_labels_estimation
from respy.solve import solve_with_backward_induction
from respy.state_space import StateSpace


def get_crit_func(params, options, df, version="log_like"):
    """Get the criterion function.

    Return a version of the likelihood functions in respy where all arguments
    except the parameter vector are fixed with :func:`functools.partial`. Thus the
    function can be directly passed into an optimizer or a function for taking
    numerical derivatives.

    By default we return :func:`log_like`. Other versions can be requested via the
    version argument.

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame containing model parameters.
    options : dict
        Dictionary containing model options.
    df : pandas.DataFrame
        The model is fit to this dataset.
    version : str, default "log_like"
        Can take the values "log_like" and "log_like_obs".

    Returns
    -------
    criterion_function : :func:`log_like`
        Criterion function where all arguments except the parameter vector are set.

    Raises
    ------
    AssertionError
        If data has not the expected format.

    """
    optim_paras, options = process_params_and_options(params, options)

    optim_paras = _adjust_optim_paras_for_estimation(optim_paras, df)

    check_estimation_data(df, optim_paras)

    state_space = StateSpace(optim_paras, options)

    (
        choices,
        idx_indiv_first_obs,
        indices,
        log_wages_observed,
        type_covariates,
    ) = _process_estimation_data(df, state_space, optim_paras, options)

    base_draws_est = create_base_draws(
        (len(choices), options["estimation_draws"], len(optim_paras["choices"])),
        next(options["estimation_seed_startup"]),
    )

    if version == "log_like":
        unpartialed = log_like
    elif version == "log_like_obs":
        unpartialed = log_like_obs
    else:
        raise ValueError("version has to be 'log_like' or 'log_like_obs'.")

    criterion_function = partial(
        unpartialed,
        choices=choices,
        idx_indiv_first_obs=idx_indiv_first_obs,
        indices=indices,
        log_wages_observed=log_wages_observed,
        base_draws_est=base_draws_est,
        state_space=state_space,
        type_covariates=type_covariates,
        options=options,
    )

    # this will be relevant for estimagic topography plots
    criterion_function.__name__ = version
    return criterion_function


def log_like(
    params,
    choices,
    idx_indiv_first_obs,
    indices,
    log_wages_observed,
    base_draws_est,
    state_space,
    type_covariates,
    options,
):
    """Criterion function for the likelihood maximization.

    This function calculates the average likelihood contribution of the sample.

    Parameters
    ----------
    params : pandas.Series
        Parameter Series
    choices : numpy.ndarray
        Array with shape (n_observations * n_types) containing choices for each
        individual-period pair.
    idx_indiv_first_obs : numpy.ndarray
        Array with shape (n_individuals,) containing indices for the first observation
        of each individual in the data. This is used to aggregate probabilities of the
        individual over all periods.
    indices : numpy.ndarray
        Array with shape (n_observations, n_types) containing indices to map each
        observation to its correponding state for each type.
    log_wages_observed : numpy.ndarray
        Array with shape (n_observations, n_types) containing observed log wages.
    base_draws_est : numpy.ndarray
        Set of draws to calculate the probability of observed wages.
    state_space : :class:`~respy.state_space.StateSpace`
        State space.
    options : dict
        Contains model options.

    """
    contribs = log_like_obs(
        params,
        choices,
        idx_indiv_first_obs,
        indices,
        log_wages_observed,
        base_draws_est,
        state_space,
        type_covariates,
        options,
    )

    return contribs.mean()


def log_like_obs(
    params,
    choices,
    idx_indiv_first_obs,
    indices,
    log_wages_observed,
    base_draws_est,
    state_space,
    type_covariates,
    options,
):
    """Criterion function for the likelihood maximization.

    This function calculates the likelihood contributions of the sample.

    Parameters
    ----------
    params : pandas.Series
        Parameter Series
    choices : numpy.ndarray
        Array with shape (n_observations * n_types) containing choices for each
        individual-period pair.
    idx_indiv_first_obs : numpy.ndarray
        Array with shape (n_individuals,) containing indices for the first observation
        of each individual in the data. This is used to aggregate probabilities of the
        individual over all periods.
    indices : numpy.ndarray
        Array with shape (n_observations, n_types) containing indices to map each
        observation to its correponding state for each type.
    log_wages_observed : numpy.ndarray
        Array with shape (n_observations, n_types) containing observed log wages.
    base_draws_est : numpy.ndarray
        Set of draws to calculate the probability of observed wages.
    state_space : :class:`~respy.state_space.StateSpace`
        State space.
    options : dict
        Contains model options.

    """
    optim_paras, options = process_params_and_options(params, options)

    state_space.update_systematic_rewards(optim_paras)

    state_space = solve_with_backward_induction(state_space, optim_paras, options)

    contribs = _internal_log_like_obs(
        state_space,
        choices,
        idx_indiv_first_obs,
        indices,
        log_wages_observed,
        base_draws_est,
        type_covariates,
        optim_paras,
        options,
    )

    return contribs


def _internal_log_like_obs(
    state_space,
    choices,
    idx_indiv_first_obs,
    indices,
    log_wages_observed,
    base_draws_est,
    type_covariates,
    optim_paras,
    options,
):
    """Calculate the likelihood contribution of each individual in the sample.

    The function calculates all likelihood contributions for all observations in the
    data which means all individual-period-type combinations.

    Then, likelihoods are accumulated within each individual and type over all periods.
    After that, the result is multiplied with the type-specific shares which yields the
    contribution to the likelihood for each individual.

    Parameters
    ----------
    state_space : :class:`~respy.state_space.StateSpace`
        Class of state space.
    choices : numpy.ndarray
        Array with shape (n_observations * n_types) containing choices for each
        individual-period pair.
    idx_indiv_first_obs : numpy.ndarray
        Array with shape (n_individuals,) containing indices for the first observation
        of each individual in the data. This is used to aggregate probabilities of the
        individual over all periods.
    indices : numpy.ndarray
        Array with shape (n_observations, n_types) containing indices to map each
        observation to its correponding state for each type.
    log_wages_observed : numpy.ndarray
        Array with shape (n_observations, n_types) containing observed log wages.
    base_draws_est : numpy.ndarray
        Array with shape (n_periods, n_draws, n_choices) containing i.i.d. draws from
        standard normal distributions.
    optim_paras : dict
        Dictionary with quantities that were extracted from the parameter vector.
    options : dict

    Returns
    -------
    contribs : numpy.ndarray
        Array with shape (n_individuals,) containing contributions of individuals in the
        empirical data.

    """
    n_obs, n_types = indices.shape
    n_choices = len(optim_paras["choices"])

    wages_systematic = state_space.wages[indices].reshape(n_obs * n_types, -1)

    draws, wage_loglikes = create_draws_and_log_prob_wages(
        log_wages_observed,
        wages_systematic,
        base_draws_est,
        choices,
        optim_paras["shocks_cholesky"],
        len(optim_paras["choices_w_wage"]),
        optim_paras["meas_error"],
        optim_paras["is_meas_error"],
    )

    draws = draws.reshape(n_obs, n_types, -1, n_choices)

    # Get continuation values. The problem is that we only need a subset of continuation
    # values defined in ``indices``. To not create the complete matrix of continuation
    # values, select only necessary continuation value indices.
    selected_indices = state_space.indices_of_child_states[indices]
    continuation_values = state_space.emax_value_functions[selected_indices]
    continuation_values = np.where(selected_indices >= 0, continuation_values, 0)

    choice_loglikes = simulate_log_probability_of_individuals_observed_choice(
        state_space.wages[indices],
        state_space.nonpec[indices],
        continuation_values,
        draws,
        optim_paras["delta"],
        state_space.is_inadmissible[indices],
        choices.reshape(-1, n_types),
        options["estimation_tau"],
    )

    wage_loglikes = wage_loglikes.reshape(n_obs, n_types)

    per_period_loglikes = wage_loglikes + choice_loglikes

    per_individual_loglikes = np.add.reduceat(per_period_loglikes, idx_indiv_first_obs)
    if n_types >= 2:
        z = np.dot(type_covariates, optim_paras["type_prob"].T)
        type_probabilities = softmax(z, axis=1)

        log_type_probabilities = np.log(type_probabilities)
        weighted_loglikes = per_individual_loglikes + log_type_probabilities

        contribs = logsumexp(weighted_loglikes, axis=1)
    else:
        contribs = per_individual_loglikes.flatten()

    contribs = np.clip(contribs, -HUGE_FLOAT, HUGE_FLOAT)

    return contribs


@nb.njit
def softmax_i(x, i, tau=1):
    """Calculate log probability of a soft maximum for index ``i``.

    The log softmax function is essentially

    .. math::

        log_softmax_i = log(softmax(x_i))

    This function is superior to the naive implementation as takes advantage of the log
    and uses the fact that the softmax function is shift-invariant. Integrating the log
    in the softmax function yields

    .. math::

        log_softmax_i = x_i - max(x) - log(sum(exp(x - max(x))))

    The second property ensures that overflows and underflows in the exponential
    function cannot happen as ``exp(x - max(x)) = exp(0) = 1`` at its highest value and
    ``exp(-inf) = 0`` at its lowest value. Only infinite inputs can cause an invalid
    output.

    The temperature parameter ``tau`` controls the smoothness of the function. For
    ``tau`` close to 1, the function is more similar to ``max(x)`` and the derivatives
    of the functions grow to infinity. As ``tau`` grows, the smoothed maximum is bigger
    than the actual maximum and derivatives diminish. See [1]_ for more information on
    the smoothing factor ``tau``. Note that the post covers the inverse of the smoothing
    factor in this function. It is also covered in [2]_ as the kernel-smoothing choice
    probability simulator. In reinforcement learning and statistical mechanics the
    function is known as the Boltzmann exploration.

    .. [1] https://www.johndcook.com/blog/2010/01/13/soft-maximum
    .. [2] McFadden, D. (1989). A method of simulated moments for estimation of discrete
           response models without numerical integration. Econometrica: Journal of the
           Econometric Society, 995-1026.

    Parameters
    ----------
    x : np.ndarray
        Array with shape (n,) containing the values for which a smoothed maximum should
        be computed.
    i : int
        Index for which the log probability should be computed.
    tau : float
        Smoothing parameter to control the size of derivatives.

    Returns
    -------
    log_probability : float
        Log probability for input value ``i``.

    """
    max_x = np.max(x)
    smoothed_differences = (x - max_x) / tau
    sum_exp = np.sum(np.exp(smoothed_differences))
    smoothed_probability = np.exp(smoothed_differences[i]) / sum_exp

    return smoothed_probability


@nb.guvectorize(
    ["f8[:], f8[:], f8[:], f8[:, :], f8, b1[:], i8, f8, f8[:]"],
    "(n_choices), (n_choices), (n_choices), (n_draws, n_choices), (), (n_choices), (), "
    "() -> ()",
    nopython=True,
    target="parallel",
)
def simulate_log_probability_of_individuals_observed_choice(
    wages,
    nonpec,
    continuation_values,
    draws,
    delta,
    is_inadmissible,
    choice,
    tau,
    log_smoothed_probability,
):
    """Simulate the probability of observing the agent's choice.

    The probability is simulated by iterating over a distribution of unobservables.
    First, the utility of each choice is computed. Then, the probability of observing
    the choice of the agent given the maximum utility from all choices is computed.

    Parameters
    ----------
    wages : numpy.ndarray
        Array with shape (n_choices,).
    nonpec : numpy.ndarray
        Array with shape (n_choices,).
    continuation_values : numpy.ndarray
        Array with shape (n_choices,)
    draws : numpy.ndarray
        Array with shape (n_draws, n_choices)
    delta : float
        Discount rate.
    is_inadmissible: numpy.ndarray
        Array with shape (n_choices,) containing an indicator for each choice whether
        the following state is inadmissible.
    choice : int
        Choice of the agent.
    tau : float
        Smoothing parameter for choice probabilities.

    Returns
    -------
    log_prob_choice : float
        Smoothed log probability of choice.

    """
    n_draws, n_choices = draws.shape

    value_functions = np.zeros(n_choices)

    smoothed_probability = 0

    for i in range(n_draws):

        for j in range(n_choices):
            value_function, _ = aggregate_keane_wolpin_utility(
                wages[j],
                nonpec[j],
                continuation_values[j],
                draws[i, j],
                delta,
                is_inadmissible[j],
            )

            value_functions[j] = value_function

        smoothed_probability += softmax_i(value_functions, choice, tau)

    smoothed_probability /= n_draws

    log_smoothed_probability_ = np.log(smoothed_probability)

    log_smoothed_probability[0] = log_smoothed_probability_


def _convert_choice_variables_from_categorical_to_codes(df, options):
    """Recode choices to choice codes in the model.

    We cannot use ``.cat.codes`` because order might be different. The model requires an
    order of ``choices_w_exp_w_wag``, ``choices_w_exp_wo_wage``,
    ``choices_wo_exp_wo_wage``.

    See also
    --------
    respy.pre_processing.model_processing._order_choices

    """
    choices_to_codes = {choice: i for i, choice in enumerate(options["choices"])}
    df.choice = df.choice.replace(choices_to_codes).astype(np.uint8)
    for i in range(1, options["n_lagged_choices"] + 1):
        df[f"lagged_choice_{i}"] = (
            df[f"lagged_choice_{i}"].replace(choices_to_codes).astype(np.uint8)
        )

    return df


def _process_estimation_data(df, state_space, optim_paras, options):
    """Process estimation data.

    All necessary objects for :func:`_internal_log_like_obs` dependent on the data are
    produced.

    Some objects have to be repeated for each type which is a desirable format for the
    estimation where every observations is weighted by type probabilities.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame which contains the data used for estimation. The DataFrame
        contains individual identifiers, periods, experiences, lagged choices, choices
        in current period, the wage and other observed data.
    state_space : ~respy.state_space.StateSpace
    optim_paras : dict

    Returns
    -------
    choices : numpy.ndarray
        Array with shape (n_observations, n_types) where information is only repeated
        over the second axis.
    idx_indiv_first_obs : numpy.ndarray
        Array with shape (n_individuals,) containing indices for the first observations
        of each individual.
    indices : numpy.ndarray
        Array with shape (n_observations, n_types) containing indices for states which
        correspond to observations.
    log_wages_observed : numpy.ndarray
        Array with shape (n_observations, n_types) containing clipped log wages.
    type_covariates : numpy.ndarray
        Array with shape (n_individuals, n_type_covariates) containing covariates to
        predict probabilities for each type.

    """
    labels, _ = generate_column_labels_estimation(optim_paras)

    df = df.sort_values(["Identifier", "Period"])[labels]
    df = df.rename(columns=lambda x: x.replace("Experience", "exp").lower())
    df = _convert_choice_variables_from_categorical_to_codes(df, optim_paras)

    # Get indices of states in the state space corresponding to all observations for all
    # types. The indexer has the shape (n_observations, n_types).
    indices = ()

    for period in range(df.period.max() + 1):
        period_df = df.loc[df.period.eq(period)]

        period_experience = tuple(
            period_df[col].to_numpy() for col in period_df.filter(like="exp_").columns
        )
        period_lagged_choice = tuple(
            period_df[f"lagged_choice_{i}"].to_numpy()
            for i in range(1, optim_paras["n_lagged_choices"] + 1)
        )

        period_indices = state_space.indexer[period][
            period_experience + period_lagged_choice
        ]

        indices += (period_indices,)

    indices = np.row_stack(indices)

    # The indexer is now sorted in period-individual pairs whereas the estimation needs
    # individual-period pairs. Sort it!
    indices_to_reorder = (
        df.sort_values(["period", "identifier"])
        .assign(__index__=np.arange(df.shape[0]))
        .sort_values(["identifier", "period"])["__index__"]
        .to_numpy()
    )
    indices = indices[indices_to_reorder]

    # Get an array of positions of the first observation for each individual. This is
    # used in :func:`_internal_log_like_obs` to aggregate probabilities of the
    # individual over all periods.
    n_obs_per_indiv = np.bincount(df.identifier.to_numpy())
    idx_indiv_first_obs = np.hstack((0, np.cumsum(n_obs_per_indiv)[:-1]))

    # For the estimation, log wages are needed with shape (n_observations, n_types).
    log_wages_observed = (
        np.log(df.wage.to_numpy())
        .clip(-HUGE_FLOAT, HUGE_FLOAT)
        .repeat(optim_paras["n_types"])
    )

    # For the estimation, choices are needed with shape (n_observations * n_types).
    choices = df.choice.to_numpy().repeat(optim_paras["n_types"])

    # For the type covariates, we only need the first observation of each individual.
    states = df.groupby("identifier").first()
    type_covariates = (
        create_type_covariates(states, optim_paras, options)
        if optim_paras["n_types"] > 1
        else None
    )

    return choices, idx_indiv_first_obs, indices, log_wages_observed, type_covariates


def _adjust_optim_paras_for_estimation(optim_paras, df):
    """Adjust optim_paras for estimation.

    There are some option values which are necessary for the simulation, but they can be
    directly inferred from the data for estimation. A warning is raised for the user
    which can be suppressed by adjusting the optim_paras.

    """
    for choice in optim_paras["choices_w_exp"]:

        # Adjust initial experience levels for all choices with experiences.
        init_exp_data = np.sort(
            df.loc[df.Period.eq(0), f"Experience_{choice.title()}"].unique()
        )
        init_exp_options = optim_paras["choices"][choice]["start"]
        if not np.array_equal(init_exp_data, init_exp_options):
            warnings.warn(
                f"The initial experience for choice '{choice}' differs between data, "
                f"{init_exp_data}, and optim_paras, {init_exp_options}. The optim_paras"
                " are ignored.",
                category=UserWarning,
            )
            optim_paras["choices"][choice]["start"] = init_exp_data
            optim_paras["choices"][choice].pop("share")
            optim_paras = {
                k: v
                for k, v in optim_paras.items()
                if not k.startswith("lagged_choice_")
            }

    return optim_paras
