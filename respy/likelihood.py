from functools import partial

import numpy as np
import pandas as pd
from numba import guvectorize

from respy.conditional_draws import create_draws_and_prob_wages
from respy.config import HUGE_FLOAT
from respy.pre_processing.data_checking import check_estimation_data
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import aggregate_keane_wolpin_utility
from respy.shared import clip
from respy.shared import create_base_draws
from respy.shared import predict_multinomial_logit
from respy.solve import solve_with_backward_induction
from respy.state_space import create_base_covariates
from respy.state_space import StateSpace


def get_crit_func(params, options, df):
    """Get the criterion function.

    The return value is the :func:`log_like` where all arguments except the parameter
    vector are fixed with :func:`functools.partial`. Thus, the function can be passed
    directly into any optimization algorithm.

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame containing model parameters.
    options : dict
        Dictionary containing model options.
    df : pandas.DataFrame
        The model is fit to this dataset.

    Returns
    -------
    criterion_function : :func:`log_like`
        Criterion function where all arguments except the parameter vector are set.

    Raises
    ------
    AssertionError
        If data has not the expected format.

    """
    params, optim_paras, options = process_params_and_options(params, options)

    check_estimation_data(df, options)

    df = _process_estimation_data(df, options)

    state_space = StateSpace(params, options)

    base_draws_est = create_base_draws(
        (options["n_periods"], options["estimation_draws"], len(options["choices"])),
        options["estimation_seed"],
    )

    # For the type covariates, we only need the first observation of each individual.
    states = df.copy().groupby("identifier").first()
    type_covariates = (
        create_type_covariates(states, options) if options["n_types"] > 1 else None
    )

    criterion_function = partial(
        log_like,
        data=df,
        base_draws_est=base_draws_est,
        state_space=state_space,
        type_covariates=type_covariates,
        options=options,
    )
    return criterion_function


def log_like(params, data, base_draws_est, state_space, type_covariates, options):
    """Criterion function for the likelihood maximization.

    This function calculates the average likelihood contribution of the sample.

    Parameters
    ----------
    params : pandas.Series
        Parameter Series
    data : pandas.DataFrame
        The log likelihood is calculated for this data.
    base_draws_est : numpy.ndarray
        Set of draws to calculate the probability of observed wages.
    state_space : :class:`~respy.state_space.StateSpace`
        State space.
    options : dict
        Contains model options.

    """
    params, optim_paras, options = process_params_and_options(params, options)

    state_space.update_systematic_rewards(optim_paras, options)

    state_space = solve_with_backward_induction(state_space, optim_paras, options)

    contribs = log_like_obs(
        state_space, data, base_draws_est, type_covariates, optim_paras, options
    )

    crit_val = -np.mean(contribs)

    return crit_val


@guvectorize(
    ["f8[:], f8[:], f8[:], f8[:, :], f8, b1[:], i8, f8, f8[:]"],
    "(n_choices), (n_choices), (n_choices), (n_draws, n_choices), (), (n_choices), (), "
    "() -> ()",
    nopython=True,
    target="parallel",
)
def simulate_probability_of_individuals_observed_choice(
    wages,
    nonpec,
    continuation_values,
    draws,
    delta,
    is_inadmissible,
    choice,
    tau,
    prob_choice,
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
    prob_choice : float
        Smoothed probability of choice.

    """
    n_draws, n_choices = draws.shape

    value_functions = np.zeros((n_choices, n_draws))

    prob_choice[0] = 0.0

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

        prob_choice[0] += value_functions[choice, i] / sum_smooth_values

    prob_choice[0] /= n_draws


def log_like_obs(
    state_space, df, base_draws_est, type_covariates, optim_paras, options
):
    """Calculate the likelihood contribution of each individual in the sample.

    The function calculates all likelihood contributions for all observations in the
    data which means all individual-period-type combinations. Then, likelihoods are
    accumulated within each individual and type over all periods. After that, the result
    is multiplied with the type-specific shares which yields the contribution to the
    likelihood for each individual.

    Parameters
    ----------
    state_space : :class:`~respy.state_space.StateSpace`
        Class of state space.
    df : pandas.DataFrame
        DataFrame with the empirical dataset.
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
    # Convert data to NumPy arrays.
    periods = df.period.to_numpy()
    lagged_choices = df.lagged_choice.to_numpy()
    choices = df.choice.to_numpy()
    experiences = tuple(df[col].to_numpy() for col in df.filter(like="exp_").columns)
    wages_observed = df.wage.to_numpy()

    # Get the number of observations for each individual and an array with indices of
    # each individual's first observation. After that, extract initial education levels
    # per agent which are important for type-specific probabilities.
    n_obs_per_indiv = np.bincount(df.identifier.to_numpy())
    idx_indiv_first_obs = np.hstack((0, np.cumsum(n_obs_per_indiv)[:-1]))

    # Get indices of states in the state space corresponding to all observations for all
    # types. The indexer has the shape (n_obs, n_types).
    ks = state_space.indexer[(periods,) + experiences + (lagged_choices,)]
    n_obs, n_types = ks.shape

    wages_observed = wages_observed.repeat(n_types)
    log_wages_observed = np.clip(np.log(wages_observed), -HUGE_FLOAT, HUGE_FLOAT)
    wages_systematic = state_space.wages[ks].reshape(n_obs * n_types, -1)
    n_choices = wages_systematic.shape[1]
    choices = choices.repeat(n_types)
    periods = state_space.states[ks, 0].flatten()

    draws, prob_wages = create_draws_and_prob_wages(
        log_wages_observed,
        wages_systematic,
        base_draws_est,
        choices,
        optim_paras["shocks_cholesky"],
        optim_paras["meas_error"],
        periods,
        len(options["choices_w_wage"]),
    )

    draws = draws.reshape(n_obs, n_types, -1, n_choices)

    prob_choices = simulate_probability_of_individuals_observed_choice(
        state_space.wages[ks],
        state_space.nonpec[ks],
        state_space.continuation_values[ks],
        draws,
        optim_paras["delta"],
        state_space.is_inadmissible[ks],
        choices.reshape(-1, n_types),
        options["estimation_tau"],
    )

    prob_obs = prob_choices * prob_wages.reshape(n_obs, -1)

    # Accumulate the likelihood of observations for each individual-type combination
    # over all periods.
    prob_type = np.multiply.reduceat(prob_obs, idx_indiv_first_obs)

    # Calculate the probabilities for all types based on an individual's initial
    # characteristics.
    type_probabilities = (
        predict_multinomial_logit(optim_paras["type_prob"], type_covariates)
        if n_types > 1
        else 1
    )

    # Multiply each individual-type contribution with its type-specific shares and sum
    # over types to get the likelihood contribution for each individual.
    contribs = (prob_type * type_probabilities).sum(axis=1)

    contribs = np.clip(np.log(contribs), -HUGE_FLOAT, HUGE_FLOAT)

    return contribs


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
    df.lagged_choice = df.lagged_choice.replace(choices_to_codes).astype(np.uint8)

    return df


def create_type_covariates(df, options):
    """Create covariates to predict type probabilities.

    In the simulation, the covariates are needed to predict type probabilities and
    assign types to simulated individuals. In the estimation, covariates are necessary
    to weight the probability of observations by type probabilities.

    """
    covariates = create_base_covariates(df, options["covariates"])

    all_data = pd.concat([covariates, df], axis="columns", sort=False)

    return all_data[options["type_covariates"]].to_numpy()


def _process_estimation_data(df, options):
    df = df.sort_values(["Identifier", "Period"])[
        ["Identifier", "Period"]
        + [f"Experience_{choice.title()}" for choice in options["choices_w_exp"]]
        + ["Lagged_Choice", "Choice", "Wage"]
    ]
    df = df.rename(columns=lambda x: x.replace("Experience", "exp").lower())

    df = _convert_choice_variables_from_categorical_to_codes(df, options)

    return df
