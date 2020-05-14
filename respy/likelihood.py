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
from respy.parallelization import parallelize_across_dense_dimensions
from respy.parallelization import split_and_combine_df
from respy.pre_processing.data_checking import check_estimation_data
from respy.pre_processing.model_processing import process_params_and_options
from respy.pre_processing.process_covariates import identify_necessary_covariates
from respy.shared import aggregate_keane_wolpin_utility
from respy.shared import compute_covariates
from respy.shared import convert_labeled_variables_to_codes
from respy.shared import create_base_draws
from respy.shared import downcast_to_smallest_dtype
from respy.shared import generate_column_dtype_dict_for_estimation
from respy.shared import map_observations_to_states
from respy.shared import pandas_dot
from respy.shared import rename_labels_to_internal
from respy.solve import get_solve_func


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

    optim_paras = _update_optim_paras_with_initial_experience_levels(optim_paras, df)

    check_estimation_data(df, optim_paras)

    solve = get_solve_func(params, options)
    state_space = solve.keywords["state_space"]

    df, type_covariates = _process_estimation_data(
        df, state_space, optim_paras, options
    )

    # Replace with decorator.
    base_draws_est = {}
    for dense_index, indices in df.groupby("dense_index").groups.items():
        n_choices = sum(state_space.dense_index_to_choice_set[dense_index])
        draws = create_base_draws(
            (len(indices), options["estimation_draws"], n_choices),
            next(options["estimation_seed_startup"]),
            options["monte_carlo_sequence"],
        )
        base_draws_est[dense_index] = draws

    criterion_function = partial(
        log_like,
        df=df,
        base_draws_est=base_draws_est,
        solve=solve,
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
    solve,
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
    solve : :func:`~respy.solve.solve`
        Function which solves the model with new parameters.
    options : dict
        Contains model options.

    """
    optim_paras, options = process_params_and_options(params, options)

    state_space = solve(params)

    contribs, df, log_type_probabilities = _internal_log_like_obs(
        state_space, df, base_draws_est, type_covariates, optim_paras, options
    )

    # Return mean log likelihood or log likelihood contributions.
    out = contribs.mean() if return_scalar else contribs

    if return_comparison_plot_data:
        comparison_plot_data = _create_comparison_plot_data(
            df, log_type_probabilities, optim_paras
        )
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

    n_types = optim_paras["n_types"]

    wages = state_space.get_attribute("wages")
    nonpecs = state_space.get_attribute("nonpecs")
    continuation_values = {}
    for period in range(options["n_periods"]):
        continuation_values = {
            **continuation_values,
            **state_space.get_continuation_values(period),
        }

    df = _compute_wage_and_choice_likelihood_contributions(
        df, base_draws_est, wages, nonpecs, continuation_values, optim_paras, options,
    )

    # Aggregate choice probabilities and wage densities to log likes per observation.
    loglikes = (
        df.groupby(["identifier", "period", "type"])[["loglike_choice", "loglike_wage"]]
        .first()
        .unstack("type")
        if optim_paras["n_types"] >= 2
        else df[["loglike_choice", "loglike_wage"]]
    )
    per_observation_loglikes = loglikes["loglike_choice"] + loglikes["loglike_wage"]
    per_individual_loglikes = per_observation_loglikes.groupby("identifier").sum()

    if n_types >= 2:
        # To not alter the attribute in the functools.partial, create a copy.
        type_covariates = type_covariates.copy()
        # Weight each type-specific individual log likelihood with the type probability.
        log_type_probabilities = _compute_log_type_probabilities(
            type_covariates, optim_paras, options
        )
        weighted_loglikes = per_individual_loglikes + log_type_probabilities

        contribs = special.logsumexp(weighted_loglikes, axis=1)
    else:
        contribs = per_individual_loglikes.to_numpy().flatten()
        log_type_probabilities = None

    contribs = np.clip(contribs, MIN_FLOAT, MAX_FLOAT)

    return contribs, df, log_type_probabilities


@split_and_combine_df
@parallelize_across_dense_dimensions
def _compute_wage_and_choice_likelihood_contributions(
    df, base_draws_est, wages, nonpecs, continuation_values, optim_paras, options,
):
    indices = df["index"].to_numpy()

    wages_systematic = wages[indices]
    log_wages_observed = df["log_wage"].to_numpy()
    choices = df["choice"].to_numpy()

    draws, wage_loglikes = create_draws_and_log_prob_wages(
        log_wages_observed,
        wages_systematic,
        base_draws_est,
        choices,
        optim_paras["shocks_cholesky"],
        len(optim_paras["choices_w_wage"]),
        optim_paras["meas_error"],
        optim_paras["has_meas_error"],
    )

    n_choices = wages.shape[1]
    n_observations = df.shape[0]
    draws = draws.reshape(n_observations, -1, n_choices)

    selected_continuation_values = continuation_values[indices]

    choice_loglikes = _simulate_log_probability_of_individuals_observed_choice(
        wages[indices],
        nonpecs[indices],
        selected_continuation_values,
        draws,
        optim_paras["beta_delta"],
        choices,
        options["estimation_tau"],
    )

    df["loglike_choice"] = np.clip(choice_loglikes, MIN_FLOAT, MAX_FLOAT)
    df["loglike_wage"] = np.clip(wage_loglikes, MIN_FLOAT, MAX_FLOAT)

    return df


def _compute_log_type_probabilities(df, optim_paras, options):
    """Compute the log type probabilities."""
    x_betas = _compute_x_beta_for_type_probabilities(
        df, optim_paras=optim_paras, options=options
    )

    log_probabilities = x_betas - special.logsumexp(x_betas, axis=1, keepdims=True)
    log_probabilities = np.clip(log_probabilities, MIN_FLOAT, MAX_FLOAT)

    return log_probabilities


@split_and_combine_df
@parallelize_across_dense_dimensions
def _compute_x_beta_for_type_probabilities(df, optim_paras, options):
    for type_ in range(optim_paras["n_types"]):
        first_observations = df.copy().assign(type=type_)
        relevant_covariates = identify_necessary_covariates(
            optim_paras["type_prob"][type_].index, options["covariates_all"]
        )
        first_observations = compute_covariates(first_observations, relevant_covariates)

        df[type_] = pandas_dot(first_observations, optim_paras["type_prob"][type_])

    return df[range(optim_paras["n_types"])]


@nb.njit
def _logsumexp(x):
    """Compute logsumexp of `x`.

    The function does the same as the following code, but faster.

    .. code-block:: python

        log_sum_exp = np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))

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
    ["f8[:], f8[:], f8[:], f8[:, :], f8, i8, f8, f8[:]"],
    "(n_choices), (n_choices), (n_choices), (n_draws, n_choices), (), (), () -> ()",
    nopython=True,
    target="parallel",
)
def _simulate_log_probability_of_individuals_observed_choice(
    wages,
    nonpec,
    continuation_values,
    draws,
    delta,
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
                wages[j], nonpec[j], continuation_values[j], draws[i, j], delta
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
    indexer : numpy.ndarray
        Indexer for the core state space.
    optim_paras : dict
    options : dict

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
    n_types = optim_paras["n_types"]
    col_dtype = generate_column_dtype_dict_for_estimation(optim_paras)

    df = (
        df.sort_index()[list(col_dtype)[2:]]
        .rename(columns=rename_labels_to_internal)
        .rename_axis(index=rename_labels_to_internal)
    )
    df = convert_labeled_variables_to_codes(df, optim_paras)

    # Duplicate observations for each type.
    if n_types >= 2:
        df = pd.concat([df.copy().assign(type=i) for i in range(n_types)])

    df["dense_index"], df["index"] = map_observations_to_states(
        df, state_space, optim_paras
    )

    # For the estimation, log wages are needed with shape (n_observations, n_types).
    df["log_wage"] = np.log(np.clip(df.wage.to_numpy(), 1 / MAX_FLOAT, MAX_FLOAT))
    df = df.drop(columns="wage")

    # For the type covariates, we only need the first observation of each individual.
    if n_types >= 2:
        initial_states = df.query("period == 0").copy()
        type_covariates = compute_covariates(
            initial_states, options["covariates_core"], raise_errors=False
        )
        type_covariates = type_covariates.apply(downcast_to_smallest_dtype)
    else:
        type_covariates = None

    return df, type_covariates


def _update_optim_paras_with_initial_experience_levels(optim_paras, df):
    """Adjust the initial experience levels in optim_paras from the data."""
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


def _create_comparison_plot_data(df, log_type_probabilities, optim_paras):
    """Create DataFrame for estimagic's comparison plot."""
    df = df.copy()

    df["choice"] = df["choice"].replace(dict(enumerate(optim_paras["choices"])))

    # During the likelihood calculation, the log likelihood for missing wages is
    # substituted with 0. Remove these log likelihoods to get the correct picture.
    is_wage_missing = df.log_wage.isna() & df.choice.isin(optim_paras["choices_w_wage"])
    df.loc[is_wage_missing, "loglike_wage"] = np.nan

    # Keep the log likelihood and the choice.
    columns = df.filter(like="loglike").columns.tolist() + ["choice"]
    df = df[columns]

    df = df.reset_index().melt(id_vars=["identifier", "period", "choice"])

    splitted_label = df.variable.str.split("_", expand=True)
    df["kind"] = splitted_label[1]
    df = df.drop(columns="variable").dropna()

    if log_type_probabilities is not None:
        log_type_probabilities = (
            log_type_probabilities.copy()
            .reset_index()
            .melt(
                id_vars=["identifier", "period"],
                value_vars=range(optim_paras["n_types"]),
                var_name="type",
                value_name="log_type_probability",
            )
        )

        df = df.append(log_type_probabilities, sort=False)

    return df
