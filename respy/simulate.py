"""Everything related to the simulation of data with structural models."""
import functools
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax

from respy.config import DTYPE_STATES
from respy.parallelization import parallelize_across_dense_dimensions
from respy.parallelization import split_and_combine_df
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import calculate_value_functions_and_flow_utilities
from respy.shared import compute_covariates
from respy.shared import create_base_draws
from respy.shared import create_state_space_columns
from respy.shared import downcast_to_smallest_dtype
from respy.shared import map_observations_to_states
from respy.shared import pandas_dot
from respy.shared import rename_labels_from_internal
from respy.shared import rename_labels_to_internal
from respy.shared import transform_base_draws_with_cholesky_factor
from respy.solve import get_solve_func


def get_simulate_func(
    params,
    options,
    method="n_step_ahead_with_sampling",
    df=None,
    n_simulation_periods=None,
):
    """Get the simulation function.

    Return :func:`simulate` where all arguments except the parameter vector are fixed
    with :func:`functools.partial`. Thus, the function can be directly passed into an
    optimizer for estimation with simulated method of moments or other techniques.

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame containing model parameters.
    options : dict
        Dictionary containing model options.
    method : {"n_step_ahead_with_sampling", "n_step_ahead_with_data", "one_step_ahead"}
        The simulation method which can be one of three and is explained in more detail
        in :func:`simulate`.
    df : pandas.DataFrame or None
        DataFrame containing one or multiple observations per individual.
    n_simulation_periods : int or None
        Simulate data for a number of periods. This options does not affect
        ``options["n_periods"]`` which controls the number of periods for which decision
        rules are computed.

    Returns
    -------
    simulate_function : :func:`simulate`
        Simulation function where all arguments except the parameter vector are set.

    """
    optim_paras, options = process_params_and_options(params, options)

    n_simulation_periods, options = _harmonize_simulation_arguments(
        method, df, n_simulation_periods, options
    )

    df = _process_input_df_for_simulation(df, method, options, optim_paras)

    solve = get_solve_func(params, options)

    # We draw shocks for all observations and for all choices although some choices
    # might not be available. Later, only the relevant shocks are selected.
    n_observations = (
        df.shape[0]
        if method == "one_step_ahead"
        else df.shape[0] * n_simulation_periods
    )
    shape = (n_observations, len(optim_paras["choices"]))

    base_draws_sim = create_base_draws(
        shape, next(options["simulation_seed_startup"]), "random"
    )
    base_draws_wage = create_base_draws(
        shape, next(options["simulation_seed_startup"]), "random"
    )

    simulate_function = functools.partial(
        simulate,
        base_draws_sim=base_draws_sim,
        base_draws_wage=base_draws_wage,
        df=df,
        method=method,
        n_simulation_periods=n_simulation_periods,
        solve=solve,
        options=options,
    )

    return simulate_function


def simulate(
    params,
    base_draws_sim,
    base_draws_wage,
    df,
    method,
    n_simulation_periods,
    solve,
    options,
):
    """Perform a simulation.

    This function performs one of three possible simulation exercises. The type of the
    simulation is controlled by ``method`` in :func:`get_simulate_func`. Ordered from no
    data to panel data on individuals, there is:

    1. *n-step-ahead simulation with sampling*: The first observation of an individual
       is sampled from the initial conditions, i.e., the distribution of observed
       variables or initial experiences, etc. in the first period. Then, the individuals
       are guided for ``n`` periods by the decision rules from the solution of the
       model.

    2. *n-step-ahead simulation with data*: Instead of sampling individuals from the
       initial conditions, take the first observation of each individual in the data.
       Then, do as in 1..

    3. *one-step-ahead simulation*: Take the complete data and find for each observation
       the corresponding outcomes, e.g, choices and wages, using the decision rules from
       the model solution.

    Parameters
    ----------
    params : pandas.DataFrame or pandas.Series
        Contains parameters.
    base_draws_sim : numpy.ndarray
        Array with shape (n_periods, n_individuals, n_choices) to provide a unique set
        of shocks for each individual in each period.
    base_draws_wage : numpy.ndarray
        Array with shape (n_periods, n_individuals, n_choices) to provide a unique set
        of wage measurement errors for each individual in each period.
    df : pandas.DataFrame or None
        Can be one three objects:

        - :data:`None` if no data is provided. This triggers sampling from initial
          conditions and a n-step-ahead simulation.
        - :class:`pandas.DataFrame` containing panel data on individuals which triggers
          a one-step-ahead simulation.
        - :class:`pandas.DataFrame` containing only first observations which triggers a
          n-step-ahead simulation taking the data as initial conditions.
    method : str
        The simulation method.
    n_simulation_periods : int
        Number periods to simulate.
    solve : :func:`~respy.solve.solve`
        Function which creates the solution of the model with new parameters.
    options : dict
        Contains model options.

    Returns
    -------
    simulated_data : pandas.DataFrame
        DataFrame of simulated individuals.

    """
    # Copy DataFrame so that the DataFrame attached to :func:`simulate` is not altered.
    df = df.copy()
    is_n_step_ahead = method != "one_step_ahead"

    optim_paras, options = process_params_and_options(params, options)
    state_space = solve(params)

    # Prepare simulation.
    df = _extend_data_with_sampled_characteristics(df, optim_paras, options)

    # Prepare shocks and store them in the pandas.DataFrame.
    draws_wage_transformed = np.exp(base_draws_wage * optim_paras["meas_error"])

    data = []
    for period in range(n_simulation_periods):
        # If it is a one-step-ahead simulation, we pick rows from the panel data. For
        # n-step-ahead simulation, `df` always contains only data of the current period.
        current_df = df.query("period == @period").copy()

        if method == "one_step_ahead":
            slice_ = np.where(df.eval("period == @period"))[0]
        else:
            slice_ = slice(df.shape[0] * period, df.shape[0] * (period + 1))

        for i, choice in enumerate(optim_paras["choices"]):
            current_df[f"shock_reward_{choice}"] = base_draws_sim[slice_, i]
            current_df[f"meas_error_wage_{choice}"] = draws_wage_transformed[slice_, i]

        current_df["dense_index"], current_df["index"] = map_observations_to_states(
            current_df, state_space, optim_paras
        )

        wages = state_space.get_attribute_from_period("wages", period)
        nonpecs = state_space.get_attribute_from_period("nonpecs", period)
        index_to_choice_set = state_space.get_attribute_from_period(
            "dense_index_to_choice_set", period
        )
        continuation_values = state_space.get_continuation_values(period=period)

        current_df_extended = _simulate_single_period(
            current_df,
            index_to_choice_set,
            wages,
            nonpecs,
            continuation_values,
            optim_paras=optim_paras,
        )

        data.append(current_df_extended)

        if is_n_step_ahead and period != n_simulation_periods - 1:
            df = _apply_law_of_motion(current_df_extended, optim_paras)

    simulated_data = _process_simulation_output(data, optim_paras)

    return simulated_data


def _extend_data_with_sampled_characteristics(df, optim_paras, options):
    """Sample initial observations from initial conditions.

    The function iterates over all state space dimensions and replaces NaNs with values
    sampled from initial conditions. In the case of an n-step-ahead simulation with
    sampling all state space dimensions are sampled. For the other two simulation
    methods, potential NaNs in the data are replaced with sampled characteristics.

    Characteristics are sampled regardless of the simulation type which keeps randomness
    across the types constant.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame which contains only an index for n-step-ahead simulation with
        sampling. For the other simulation methods, it contains information on
        individuals which is allowed to have missing information in the first period.
    optim_paras : dict
    options : dict

    Returns
    -------
    df : pandas.DataFrame
        A pandas DataFrame with no missings at all.

    """
    # Sample characteristics only for the first period.
    fp = df.query("period == 0").copy()
    index = fp.index

    for observable in optim_paras["observables"]:
        level_dict = optim_paras["observables"][observable]
        sampled_char = _sample_characteristic(fp, options, level_dict, use_keys=False)
        fp[observable] = fp[observable].fillna(
            pd.Series(data=sampled_char, index=index), downcast="infer"
        )

    for choice in optim_paras["choices_w_exp"]:
        level_dict = optim_paras["choices"][choice]["start"]
        sampled_char = _sample_characteristic(fp, options, level_dict, use_keys=True)
        fp[f"exp_{choice}"] = fp[f"exp_{choice}"].fillna(
            pd.Series(data=sampled_char, index=index), downcast="infer"
        )

    for lag in reversed(range(1, optim_paras["n_lagged_choices"] + 1)):
        level_dict = optim_paras[f"lagged_choice_{lag}"]
        sampled_char = _sample_characteristic(fp, options, level_dict, use_keys=False)
        fp[f"lagged_choice_{lag}"] = fp[f"lagged_choice_{lag}"].fillna(
            pd.Series(data=sampled_char, index=index), downcast="infer"
        )

    # Sample types and map them to individuals for all periods.
    if optim_paras["n_types"] >= 2:
        level_dict = optim_paras["type_prob"]
        types = _sample_characteristic(fp, options, level_dict, use_keys=False)
        fp["type"] = fp["type"].fillna(
            pd.Series(data=types, index=index), downcast="infer"
        )

    # Update data in the first period with sampled characteristics.
    df = df.combine_first(fp)

    # Types are invariant and we have to fill the DataFrame for one-step-ahead.
    if optim_paras["n_types"] >= 2:
        df["type"] = df["type"].fillna(method="ffill", downcast="infer")

    state_space_columns = create_state_space_columns(optim_paras)
    df = df[state_space_columns].astype(DTYPE_STATES)

    return df


@split_and_combine_df
@parallelize_across_dense_dimensions
def _simulate_single_period(
    df, choice_set, wages, nonpecs, continuation_values, optim_paras
):
    """Simulate individuals in a single period.

    The function performs the following sets:

    - Map individuals in one period to the states in the model.
    - Simulate choices and wages for those individuals.
    - Store additional information in a :class:`pandas.DataFrame` and return it.

    UNtil now this function assumes that there are no mixed constraints.
    See docs for more infomration!
    """
    valid_choices = [x for i, x in enumerate(optim_paras["choices"]) if choice_set[i]]

    n_wages_raw = len(optim_paras["choices_w_wage"])
    n_wages = sum(choice_set[:n_wages_raw])

    # Get indices which connect states in the state space and simulated agents. Subtract
    # the minimum of indices (excluding invalid indices) because wages, etc. contain
    # only wages in this period and normal indices select rows from all wages.

    period_indices = df["index"].to_numpy()
    try:
        wages = wages[period_indices]
        nonpecs = nonpecs[period_indices]
        continuation_values = continuation_values[period_indices]
    except IndexError as e:
        raise Exception(
            "Simulated individuals could not be mapped to their corresponding states in"
            " the state space. This might be caused by a mismatch between "
            "option['core_state_space_filters'] and the initial conditions."
        ) from e

    draws_shock = df[[f"shock_reward_{c}" for c in valid_choices]].to_numpy()
    draws_shock_transformed = transform_base_draws_with_cholesky_factor(
        draws_shock, choice_set, optim_paras["shocks_cholesky"], optim_paras
    )

    draws_wage = df[[f"meas_error_wage_{c}" for c in valid_choices]].to_numpy()
    value_functions, flow_utilities = calculate_value_functions_and_flow_utilities(
        wages,
        nonpecs,
        continuation_values,
        draws_shock_transformed,
        optim_paras["beta_delta"],
    )
    choice = np.nanargmax(value_functions, axis=1)

    # Get choice replacement dict. There is too much positioning until now!
    wages = wages * draws_shock_transformed * draws_wage
    wages[:, n_wages:] = np.nan
    wage = np.choose(choice, wages.T)

    # We map choice positions to choice codes
    positions = [i for i, x in enumerate(optim_paras["choices"]) if x in valid_choices]
    for pos, val in enumerate(positions):
        choice = np.where(choice == pos, val, choice)

    # Store necessary information and information for debugging, etc..

    df["choice"] = choice
    df["wage"] = wage
    df["discount_rate"] = optim_paras["delta"]
    df["present_bias"] = optim_paras["beta"]

    for i, choice in enumerate(valid_choices):
        df[f"nonpecuniary_reward_{choice}"] = nonpecs[:, i]
        df[f"wage_{choice}"] = wages[:, i]
        df[f"flow_utility_{choice}"] = flow_utilities[:, i]
        df[f"value_function_{choice}"] = value_functions[:, i]
        df[f"continuation_value_{choice}"] = continuation_values[:, i]

    return df


def _sample_characteristic(states_df, options, level_dict, use_keys):
    """Sample characteristic of individuals.

    The function is used to sample the values of one state space characteristic, say
    experience. The keys of ``level_dict`` are the possible starting values of
    experience. The values of the dictionary are :class:`pandas.Series` whose index are
    covariate names and the values are the parameter values.

    ``states_df`` is used to generate all possible covariates with the existing
    information.

    For each level, the dot product of parameters and covariates determines the value
    ``z``. The softmax function converts the level-specific ``z``-values to
    probabilities. The probabilities are used to sample the characteristic.

    Parameters
    ----------
    states_df : pandas.DataFrame
        Contains the state of each individual.
    options : dict
        Options of the model.
    level_dict : dict
        A dictionary where the keys are the values distributed according to the
        probability mass function. The values are a :class:`pandas.Series` with
        covariate names as the index and parameter values.
    use_keys : bool
        Identifier for whether the keys of the level dict should be used as variables
        values or use numeric codes instead. For example, assign numbers to choices.

    Returns
    -------
    characteristic : numpy.ndarray
        Array with shape (n_individuals,) containing sampled values.

    """
    # Generate covariates.
    all_data = compute_covariates(
        states_df, options["covariates_all"], check_nans=True, raise_errors=False
    )

    # Calculate dot product of covariates and parameters.
    z = ()
    for level in level_dict:
        x_beta = pandas_dot(all_data, level_dict[level])
        z += (x_beta,)

    # Calculate probabilities with the softmax function.
    probabilities = softmax(np.column_stack(z), axis=1)

    np.random.seed(next(options["simulation_seed_iteration"]))

    choices = level_dict if use_keys else len(level_dict)
    characteristic = _random_choice(choices, probabilities)

    return characteristic


def _convert_codes_to_original_labels(df, optim_paras):
    """Convert codes in choice-related and observed variables to labels."""
    code_to_choice = dict(enumerate(optim_paras["choices"]))

    for choice_var in ["Choice"] + [
        f"Lagged_Choice_{i}" for i in range(1, optim_paras["n_lagged_choices"] + 1)
    ]:
        df[choice_var] = (
            df[choice_var]
            .astype("category")
            .cat.set_categories(code_to_choice)
            .cat.rename_categories(code_to_choice)
        )

    for observable in optim_paras["observables"]:
        code_to_obs = dict(enumerate(optim_paras["observables"][observable]))
        df[f"{observable.title()}"] = df[f"{observable.title()}"].replace(code_to_obs)

    return df


def _process_simulation_output(data, optim_paras):
    """Create simulated data.

    This function takes an array of simulated outcomes and additional information for
    each period and stacks them together to one DataFrame.

    Parameters
    ----------
    data : list
        List of DataFrames for each simulated period with internal codes and labels.
    optim_paras : dict

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with simulated data.

    """
    df = (
        pd.concat(data, sort=False)
        .sort_index()
        .rename(columns=rename_labels_from_internal)
        .rename_axis(index=rename_labels_from_internal)
    )
    df = _convert_codes_to_original_labels(df, optim_paras)

    # We use the downcast to convert some variables to integers.
    df = df.apply(downcast_to_smallest_dtype)

    return df


def _random_choice(choices, probabilities=None, decimals=5):
    """Return elements of choices for a two-dimensional array of probabilities.

    It is assumed that probabilities are ordered (n_samples, n_choices).

    The function is taken from this `StackOverflow post
    <https://stackoverflow.com/questions/40474436>`_ as a workaround for
    :func:`numpy.random.choice` as it can only handle one-dimensional probabilities.

    Example
    -------
    Here is an example with non-zero probabilities.

    >>> n_samples = 100_000
    >>> n_choices = 3
    >>> p = np.array([0.15, 0.35, 0.5])
    >>> ps = np.tile(p, (n_samples, 1))
    >>> choices = _random_choice(n_choices, ps)
    >>> np.round(np.bincount(choices), decimals=-3) / n_samples
    array([0.15, 0.35, 0.5 ])

    Here is an example where one choice has probability zero.

    >>> choices = np.arange(3)
    >>> p = np.array([0.4, 0, 0.6])
    >>> ps = np.tile(p, (n_samples, 1))
    >>> choices = _random_choice(3, ps)
    >>> np.round(np.bincount(choices), decimals=-3) / n_samples
    array([0.4, 0. , 0.6])

    """
    if isinstance(choices, int):
        choices = np.arange(choices)
    elif isinstance(choices, (dict, list, tuple)):
        choices = np.array(list(choices))
    elif isinstance(choices, np.ndarray):
        pass
    else:
        raise TypeError(f"'choices' has invalid type {type(choices)}.")

    if probabilities is None:
        n_choices = choices.shape[-1]
        probabilities = np.ones((1, n_choices)) / n_choices
        probabilities = np.broadcast_to(probabilities, choices.shape)

    cumulative_distribution = probabilities.cumsum(axis=1)
    # Probabilities often do not sum to one but 0.99999999999999999.
    cumulative_distribution[:, -1] = np.round(cumulative_distribution[:, -1], decimals)

    if not (cumulative_distribution[:, -1] == 1).all():
        raise ValueError("Probabilities do not sum to one.")

    u = np.random.rand(cumulative_distribution.shape[0], 1)

    # Note that :func:`np.argmax` returns the first index for multiple maximum values.
    indices = (u < cumulative_distribution).argmax(axis=1)

    out = np.take(choices, indices)
    if out.shape == (1,):
        out = out[0]

    return out


def _apply_law_of_motion(df, optim_paras):
    """Apply the law of motion to get the states in the next period.

    For n-step-ahead simulations, the states of the next period are generated from the
    current states and the current decision. This function changes experiences and
    previous choices according to the choice in the current period, to get the states of
    the next period.

    We implicitly assume that observed variables are constant.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame contains the simulated information of individuals in one period.
    optim_paras : dict

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame contains the states of individuals in the next period.

    """
    df = df.copy()
    n_lagged_choices = optim_paras["n_lagged_choices"]

    # Update work experiences.
    for i, choice in enumerate(optim_paras["choices_w_exp"]):
        df[f"exp_{choice}"] += df["choice"] == i

    # Update lagged choices by deleting oldest lagged, renaming other lags and inserting
    # choice in the first position.
    if n_lagged_choices:
        # Save position of first lagged choice.
        position = df.columns.tolist().index("lagged_choice_1")

        # Drop oldest lag.
        df = df.drop(columns=f"lagged_choice_{n_lagged_choices}")

        # Rename newer lags
        rename_lagged_choices = {
            f"lagged_choice_{i}": f"lagged_choice_{i + 1}"
            for i in range(1, n_lagged_choices)
        }
        df = df.rename(columns=rename_lagged_choices)

        # Add current choice as new lag.
        df.insert(position, "lagged_choice_1", df["choice"])

    # Increment period in MultiIndex by one.
    df.index = df.index.set_levels(
        df.index.get_level_values("period") + 1, level="period", verify_integrity=False
    )

    state_space_columns = create_state_space_columns(optim_paras)
    df = df[state_space_columns]

    return df


def _harmonize_simulation_arguments(method, df, n_simulation_periods, options):
    """Harmonize the arguments of the simulation.

    This function handles the interaction of the four inputs and aligns the number of
    simulated individuals and the number of simulated periods.

    """
    if n_simulation_periods is None and method == "one_step_ahead":
        n_simulation_periods = int(df.index.get_level_values("Period").max() + 1)
    else:
        n_simulation_periods = options["n_periods"]

    if method == "n_step_ahead_with_sampling":
        pass
    else:
        options["simulation_agents"] = df.index.get_level_values("Identifier").nunique()

    if options["n_periods"] < n_simulation_periods:
        options["n_periods"] = n_simulation_periods
        warnings.warn(
            f"The number of periods in the model, {options['n_periods']}, is lower "
            f"than the requested number of simulated periods, {n_simulation_periods}. "
            "Set model periods equal to simulated periods. To silence the warning, "
            "adjust your specification."
        )

    return n_simulation_periods, options


def _process_input_df_for_simulation(df, method, options, optim_paras):
    """Process a :class:`pandas.DataFrame` provided by the user for the simulation."""
    if method == "n_step_ahead_with_sampling":
        ids = np.arange(options["simulation_agents"])
        index = pd.MultiIndex.from_product((ids, [0]), names=["identifier", "period"])
        df = pd.DataFrame(index=index)

    elif method == "n_step_ahead_with_data":
        ids = np.arange(options["simulation_agents"])
        index = pd.MultiIndex.from_product((ids, [0]), names=["identifier", "period"])
        df = (
            df.copy()
            .rename(columns=rename_labels_to_internal)
            .rename_axis(index=rename_labels_to_internal)
            .reindex(index=index)
            .sort_index()
        )

    elif method == "one_step_ahead":
        df = (
            df.copy()
            .rename(columns=rename_labels_to_internal)
            .rename_axis(index=rename_labels_to_internal)
            .sort_index()
        )

    else:
        raise NotImplementedError

    state_space_columns = create_state_space_columns(optim_paras)
    df = df.reindex(columns=state_space_columns)

    # Perform two checks for NaNs.

    data = df.query("period == 0").drop(columns="type", errors="ignore")
    has_nans_in_first_period = np.any(data.isna())
    if has_nans_in_first_period and method == "n_step_ahead_with_data":
        warnings.warn(
            "The data contains 'NaNs' in the first period which are replaced with "
            "characteristics implied by the initial conditions. Fix the data to silence"
            " the warning."
        )

    has_nans = np.any(df.drop(columns="type", errors="ignore").isna())
    if has_nans and method == "one_step_ahead":
        raise ValueError(
            "The data for one-step-ahead simulation must not contain NaNs."
        )

    return df
