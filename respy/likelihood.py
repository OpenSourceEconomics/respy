"""Everything related to the estimation with maximum likelihood."""
import warnings
from functools import partial

import numba as nb
import numpy as np
import pandas as pd
from scipy import special

from respy.conditional_draws import create_draws_and_log_prob_wages
from respy.config import MAX_FLOAT
from respy.config import MIN_FLOAT
from respy.pre_processing.data_checking import check_estimation_data
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import aggregate_keane_wolpin_utility
from respy.shared import convert_labeled_variables_to_codes
from respy.shared import create_base_covariates
from respy.shared import create_base_draws
from respy.shared import create_core_state_space_columns
from respy.shared import downcast_to_smallest_dtype
from respy.shared import generate_column_dtype_dict_for_estimation
from respy.shared import rename_labels
from respy.solve import solve_with_backward_induction
from respy.state_space import StateSpace


def get_crit_func(
    params, options, df, return_scalar=True, return_comparison_plot_data=False
):
    """Get the criterion function.

    Return a version of the likelihood functions in respy where all arguments
    except the parameter vector are fixed with :func:`functools.partial`. Thus the
    function can be directly passed into an optimizer or a function for taking
    numerical derivatives.

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame containing model parameters.
    options : dict
        Dictionary containing model options.
    df : pandas.DataFrame
        The model is fit to this dataset.
    return_scalar : bool, default False
        Indicator for whether the mean log likelihood should be returned or the log
        likelihood contributions.
    return_comparison_plot_data : bool, default False
        Indicator for whether a :class:`pandas.DataFrame` with various contributions for
        the visualization with estimagic should be returned.

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

    df, type_covariates = _process_estimation_data(
        df, state_space, optim_paras, options
    )

    base_draws_est = create_base_draws(
        (
            df.shape[0] * optim_paras["n_types"],
            options["estimation_draws"],
            len(optim_paras["choices"]),
        ),
        next(options["estimation_seed_startup"]),
        options["monte_carlo_sequence"],
    )

    criterion_function = partial(
        log_like,
        df=df,
        base_draws_est=base_draws_est,
        state_space=state_space,
        type_covariates=type_covariates,
        options=options,
        return_scalar=return_scalar,
        return_comparison_plot_data=return_comparison_plot_data,
    )

    return criterion_function


def log_like(
    params,
    df,
    base_draws_est,
    state_space,
    type_covariates,
    options,
    return_scalar,
    return_comparison_plot_data,
):
    """Criterion function for the likelihood maximization.

    This function calculates the likelihood contributions of the sample.

    Parameters
    ----------
    params : pandas.Series
        Parameter Series
    df : pandas.DataFrame
        The DataFrame contains choices, log wages, the indices of the states for the
        different types.
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

    contribs, df = _internal_log_like_obs(
        state_space, df, base_draws_est, type_covariates, optim_paras, options
    )

    out = contribs.mean() if return_scalar else contribs
    if return_comparison_plot_data:
        comparison_plot_data = _create_comparison_plot_data(df, optim_paras)
        out = (out, comparison_plot_data)

    return out


def _internal_log_like_obs(
    state_space, df, base_draws_est, type_covariates, optim_paras, options
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
    df : pandas.DataFrame
        The DataFrame contains choices, log wages, the indices of the states for the
        different types.
    base_draws_est : numpy.ndarray
        Array with shape (n_periods, n_draws, n_choices) containing i.i.d. draws from
        standard normal distributions.
    type_covariates : pandas.DataFrame or None
        If the model includes types, this is a :class:`pandas.DataFrame` containing the
        covariates to compute the type probabilities.
    optim_paras : dict
        Dictionary with quantities that were extracted from the parameter vector.
    options : dict
        Options of the model.

    Returns
    -------
    contribs : numpy.ndarray
        Array with shape (n_individuals,) containing contributions of individuals in the
        empirical data.
    df : pandas.DataFrame
        Contains log wages, choices and

    """
    df = df.copy()

    n_choices = len(optim_paras["choices"])
    n_obs = df.shape[0]
    n_types = optim_paras["n_types"]

    indices = df[[f"index_type_{i}" for i in range(optim_paras["n_types"])]].to_numpy()

    wages_systematic = state_space.wages[indices].reshape(n_obs * n_types, n_choices)
    log_wages_observed = df["log_wage"].to_numpy().repeat(n_types)
    choices = df["choice"].to_numpy().repeat(n_types)

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

    choice_loglikes = _simulate_log_probability_of_individuals_observed_choice(
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

    choice_loglikes = np.clip(choice_loglikes, MIN_FLOAT, MAX_FLOAT)
    wage_loglikes = np.clip(wage_loglikes, MIN_FLOAT, MAX_FLOAT)

    choice_cols = [f"loglike_choice_type_{i}" for i in range(n_types)]
    wage_cols = [f"loglike_wage_type_{i}" for i in range(n_types)]

    df = df.reindex(columns=df.columns.tolist() + choice_cols + wage_cols)
    df[choice_cols] = choice_loglikes
    df[wage_cols] = wage_loglikes

    data = df[choice_cols].to_numpy() + df[wage_cols].to_numpy()
    per_individual_loglikes = (
        pd.DataFrame(data=data, index=df.index).groupby("identifier").sum()
    )

    if n_types >= 2:
        z = ()

        for level in optim_paras["type_prob"]:
            labels = optim_paras["type_prob"][level].index
            x_beta = np.dot(type_covariates[labels], optim_paras["type_prob"][level])

            z += (x_beta,)

        type_probabilities = special.softmax(np.column_stack(z), axis=1)

        type_probabilities = np.clip(type_probabilities, 1 / MAX_FLOAT, None)
        log_type_probabilities = np.log(type_probabilities)

        weighted_loglikes = per_individual_loglikes + log_type_probabilities

        contribs = special.logsumexp(weighted_loglikes, axis=1)
    else:
        contribs = per_individual_loglikes.to_numpy().flatten()

    contribs = np.clip(contribs, MIN_FLOAT, MAX_FLOAT)

    return contribs, df


@nb.njit
def _logsumexp(x):
    """Compute logsumexp of `x`.

    The function does the same as the following code, but faster.

    .. code-block:: python

        max_x = np.max(x)
        differences = x - max_x
        log_sum_exp = max_x + np.log(np.sum(np.exp(differences)))

    The subtraction of the maximum prevents overflows and mitigates the impact of
    underflows.

    """
    # Search maximum.
    max_x = None
    length = len(x)
    for i in range(length):
        if max_x is None or x[i] > max_x:
            max_x = x[i]

    # Calculate sum of exponential differences.
    sum_exp = 0
    for i in range(length):
        diff = x[i] - max_x
        sum_exp += np.exp(diff)

    log_sum_exp = max_x + np.log(sum_exp)

    return log_sum_exp


@nb.guvectorize(
    ["f8[:], f8[:], f8[:], f8[:, :], f8, b1[:], i8, f8, f8[:]"],
    "(n_choices), (n_choices), (n_choices), (n_draws, n_choices), (), (n_choices), (), "
    "() -> ()",
    nopython=True,
    target="parallel",
)
def _simulate_log_probability_of_individuals_observed_choice(
    wages,
    nonpec,
    continuation_values,
    draws,
    delta,
    is_inadmissible,
    choice,
    tau,
    smoothed_log_probability,
):
    r"""Simulate the probability of observing the agent's choice.

    The probability is simulated by iterating over a distribution of unobservables.
    First, the utility of each choice is computed. Then, the probability of observing
    the choice of the agent given the maximum utility from all choices is computed.

    The naive implementation calculates the log probability for choice `i` with the
    softmax function.

    .. math::

        \log(\text{softmax}(x)_i) = \log\left(
            \frac{e^{x_i}}{\sum^J e^{x_j}}
        \right)

    The following function is numerically more robust. The derivation with the two
    consecutive `logsumexp` functions is included in `#278
    <https://github.com/OpenSourceEconomics/respy/pull/288>`_.

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
    smoothed_log_probability : float
        Simulated Smoothed log probability of choice.

    """
    n_draws, n_choices = draws.shape

    smoothed_log_probabilities = np.empty(n_draws)
    smoothed_value_functions = np.empty(n_choices)

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

            smoothed_value_functions[j] = value_function / tau

        smoothed_log_probabilities[i] = smoothed_value_functions[choice] - _logsumexp(
            smoothed_value_functions
        )

    smoothed_log_prob = _logsumexp(smoothed_log_probabilities) - np.log(n_draws)

    smoothed_log_probability[0] = smoothed_log_prob


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
    col_dtype = generate_column_dtype_dict_for_estimation(optim_paras)

    df = df.sort_index()[list(col_dtype)[2:]]
    df = df.rename(columns=rename_labels).rename_axis(index=rename_labels)
    df = convert_labeled_variables_to_codes(df, optim_paras)

    # Get indices of states in the state space corresponding to all observations for all
    # types. The indexer has the shape (n_observations, n_types).
    n_periods = int(df.index.get_level_values("period").max() + 1)
    indices = ()

    for period in range(n_periods):
        period_df = df.query("period == @period")

        core_columns = create_core_state_space_columns(optim_paras)
        period_core = tuple(period_df[col].to_numpy() for col in core_columns)

        period_observables = tuple(
            period_df[observable].to_numpy()
            for observable in optim_paras["observables"]
        )

        period_indices = state_space.indexer[period][period_core + period_observables]
        indices += (period_indices,)

    indices = np.concatenate(indices).reshape(-1, optim_paras["n_types"])

    # The indexer is now sorted in period-individual pairs whereas the estimation needs
    # individual-period pairs. Sort it!
    indices_to_reorder = (
        df.sort_values(["period", "identifier"])
        .assign(__index__=np.arange(df.shape[0]))
        .sort_values(["identifier", "period"])["__index__"]
        .to_numpy()
    )
    indices = indices[indices_to_reorder]

    # Finally, add the indices to the DataFrame.
    type_index_cols = [f"index_type_{i}" for i in range(optim_paras["n_types"])]
    df = df.reindex(columns=df.columns.tolist() + type_index_cols)
    df[type_index_cols] = indices

    # For the estimation, log wages are needed with shape (n_observations, n_types).
    df["log_wage"] = np.log(np.clip(df.wage.to_numpy(), 1 / MAX_FLOAT, MAX_FLOAT))
    df = df.drop(columns="wage")

    # For the type covariates, we only need the first observation of each individual.
    if optim_paras["n_types"] >= 2:
        initial_states = df.query("period == 0")
        covariates = create_base_covariates(
            initial_states, options["covariates"], raise_errors=False
        )

        all_data = pd.concat([covariates, initial_states], axis="columns")

        type_covariates = all_data[optim_paras["type_covariates"]].apply(
            downcast_to_smallest_dtype
        )
    else:
        type_covariates = None

    return df, type_covariates


def _adjust_optim_paras_for_estimation(optim_paras, df):
    """Adjust optim_paras for estimation.

    There are some option values which are necessary for the simulation, but they can be
    directly inferred from the data for estimation. A warning is raised for the user
    which can be suppressed by adjusting the optim_paras.

    """
    for choice in optim_paras["choices_w_exp"]:

        # Adjust initial experience levels for all choices with experiences.
        init_exp_data = np.sort(
            df.query("Period == 0")[f"Experience_{choice.title()}"].unique()
        )
        init_exp_params = np.array(list(optim_paras["choices"][choice]["start"]))
        if not np.array_equal(init_exp_data, init_exp_params):
            warnings.warn(
                f"The initial experience(s) for choice '{choice}' differs between data,"
                f" {init_exp_data}, and parameters, {init_exp_params}. The parameters"
                " are ignored.",
                category=UserWarning,
            )
            optim_paras["choices"][choice]["start"] = init_exp_data
            optim_paras = {
                k: v
                for k, v in optim_paras.items()
                if not k.startswith("lagged_choice_")
            }

    return optim_paras


def _create_comparison_plot_data(df, optim_paras):
    """Create DataFrame for estimagic's comparison plot."""
    # During the likelihood calculation, the log likelihood for missing wages is
    # substituted with 0. Remove these log likelihoods to get the correct picture.
    df = df.loc[df.log_wage.notna()]

    # Keep the log likelihood and the choice.
    columns = df.filter(like="loglike").columns.tolist() + ["choice"]
    df = df[columns]

    df["choice"] = df["choice"].replace(dict(enumerate(optim_paras["choices"])))

    df = df.reset_index().melt(id_vars=["identifier", "period", "choice"])

    splitted_label = df.variable.str.split("_", expand=True)
    df["kind"] = splitted_label[1]
    df["type"] = splitted_label[3]
    df = df.drop(columns="variable")

    return df
