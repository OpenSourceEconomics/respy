import warnings
from functools import partial

import numpy as np
from numba import guvectorize

from respy.conditional_draws import create_draws_and_log_prob_wages
from respy.config import HUGE_FLOAT
from respy.pre_processing.data_checking import check_estimation_data
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import aggregate_keane_wolpin_utility
from respy.shared import clip
from respy.shared import create_base_draws
from respy.shared import create_type_covariates
from respy.shared import generate_column_labels_estimation
from respy.shared import predict_multinomial_logit
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

    state_space = StateSpace(optim_paras, options)

    check_estimation_data(df, optim_paras)

    (
        choices,
        idx_indiv_first_obs,
        indices,
        log_wages_observed,
        type_covariates,
    ) = _process_estimation_data(df, state_space, optim_paras, options)

    base_draws_est = create_base_draws(
        (
            options["n_periods"],
            options["estimation_draws"],
            len(optim_paras["choices"]),
        ),
        options["estimation_seed"],
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
        Array with shape (n_observations, n_types) containing choices for each
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
        Array with shape (n_observations, n_types) containing choices for each
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
        Array with shape (n_observations, n_types) containing choices for each
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
    periods = state_space.states[indices, 0].flatten()

    draws, wage_loglikes = create_draws_and_log_prob_wages(
        log_wages_observed,
        wages_systematic,
        base_draws_est,
        choices,
        optim_paras["shocks_cholesky"],
        periods,
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
        type_probabilities = predict_multinomial_logit(
            optim_paras["type_prob"], type_covariates
        )
        log_type_probabilities = np.log(type_probabilities)
        weighted_loglikes = per_individual_loglikes + log_type_probabilities

        # The following is equivalent to:
        # writing contribs = np.log(np.exp(weighted_loglikes).sum(axis=1))
        # but avoids overflows and underflows
        minimal_m = -700 - weighted_loglikes.min(axis=1)
        maximal_m = 700 - weighted_loglikes.max(axis=1)
        valid = minimal_m <= maximal_m
        m = np.where(valid, (minimal_m + maximal_m) / 2, np.nan).reshape(-1, 1)
        contribs = np.log(np.exp(weighted_loglikes + m).sum(axis=1)) - m.flatten()
        contribs[~valid] = -HUGE_FLOAT
    else:
        contribs = per_individual_loglikes.flatten()

    contribs = np.clip(contribs, -HUGE_FLOAT, HUGE_FLOAT)

    return contribs


@guvectorize(
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
    log_prob_choice,
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

    value_functions = np.zeros((n_choices, n_draws))

    prob_choice = 0.0

    for i in range(n_draws):

        max_value_functions = 0.0

        for j in range(n_choices):
            value_function, _ = aggregate_keane_wolpin_utility(
                wages[j],
                nonpec[j],
                continuation_values[j],
                draws[i, j],
                delta,
                is_inadmissible[j],
            )

            value_functions[j, i] = value_function

            if value_function > max_value_functions:
                max_value_functions = value_function

        sum_smooth_values = 0.0

        for j in range(n_choices):
            val_exp = np.exp((value_functions[j, i] - max_value_functions) / tau)

            val_clipped = clip(val_exp, 0.0, HUGE_FLOAT)

            value_functions[j, i] = val_clipped
            sum_smooth_values += val_clipped

        prob_choice += value_functions[choice, i] / sum_smooth_values

    prob_choice /= n_draws

    log_prob_choice[0] = np.log(prob_choice)


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

    # For the estimation, choices are needed with shape (n_observations, n_types).
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
