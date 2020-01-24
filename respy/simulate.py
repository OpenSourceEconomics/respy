"""Everything related to the simulation of data with structural models."""
import functools
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax

from respy.parallelization import distribute_and_combine_simulation
from respy.parallelization import parallelize_across_dense_dimensions
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import calculate_value_functions_and_flow_utilities
from respy.shared import compute_covariates
from respy.shared import create_base_draws
from respy.shared import create_core_state_space_columns
from respy.shared import create_state_space_columns
from respy.shared import downcast_to_smallest_dtype
from respy.shared import rename_labels_from_internal
from respy.shared import rename_labels_to_internal
from respy.shared import transform_base_draws_with_cholesky_factor
from respy.solve import solve_with_backward_induction
from respy.state_space import StateSpace


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

    df = _process_input_df_for_simulation(
        df, method, n_simulation_periods, options, optim_paras
    )

    state_space = StateSpace(optim_paras, options)

    shape = (df.shape[0], len(optim_paras["choices"]))
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
        state_space=state_space,
        options=options,
    )

    return simulate_function


def simulate(params, base_draws_sim, base_draws_wage, df, state_space, options):
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
    state_space : :class:`~respy.state_space.StateSpace`
        State space of the model.
    options : dict
        Contains model options.

    Returns
    -------
    simulated_data : pandas.DataFrame
        DataFrame of simulated individuals.

    """
    # Copy DataFrame so that the DataFrame attached to :func:`simulate` is not altered.
    df = df.copy()

    optim_paras, options = process_params_and_options(params, options)

    # Solve the model.
    state_space.create_choice_rewards(optim_paras, options)
    solve_with_backward_induction(state_space, optim_paras, options)

    # Prepare simulation.
    n_simulation_periods = int(df.index.get_level_values("period").max() + 1)

    df = _extend_data_with_sampled_characteristics(df, optim_paras, options)

    # Prepare shocks and store them in the pandas.DataFrame.
    n_wages = len(optim_paras["choices_w_wage"])
    base_draws_sim_transformed = transform_base_draws_with_cholesky_factor(
        base_draws_sim, optim_paras["shocks_cholesky"], n_wages
    )
    base_draws_wage_transformed = np.exp(base_draws_wage * optim_paras["meas_error"])
    for i, choice in enumerate(optim_paras["choices"]):
        df[f"shock_reward_{choice}"] = base_draws_sim_transformed[:, i]
        df[f"meas_error_wage_{choice}"] = base_draws_wage_transformed[:, i]

    core_columns = create_core_state_space_columns(optim_paras)
    is_n_step_ahead = np.any(df[core_columns].isna())

    data = []
    for period in range(n_simulation_periods):

        # If it is a one-step-ahead simulation, we pick rows from the panel data. For
        # n-step-ahead simulation, `df` always contains only data of the current period.
        current_df = df.query("period == @period").copy()

        current_df_extended = _simulate_single_period(
            state_space, current_df, optim_paras
        )

        data.append(current_df_extended)

        if is_n_step_ahead and period != n_simulation_periods - 1:
            next_df = _apply_law_of_motion(current_df_extended, optim_paras)
            df = df.fillna(next_df)

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
    df = df[state_space_columns]

    return df


@distribute_and_combine_simulation
@parallelize_across_dense_dimensions
def _simulate_single_period(state_space, df, optim_paras):
    """Simulate individuals in a single period.

    This function takes a set of states and simulates wages, choices and other
    information. The information is stored in a NumPy array.

    Parameter
    ---------
    state_space : :class:`~respy.state_space.StateSpace`
        State space of the model.
    df : pandas.DataFrame
        DataFrame with shape (n_individuals_in_period, n_state_space_dims) which
        contains the states of simulated individuals.
    optim_paras : dict

    """
    period = df.index.get_level_values("period").max()
    n_wages = len(optim_paras["choices_w_wage"])

    # Get indices which connect states in the state space and simulated agents.
    columns = create_core_state_space_columns(optim_paras)
    indices = state_space.indexer[period][tuple(df[col].astype(int) for col in columns)]

    # Get continuation values. Indices work on the complete state space whereas
    # continuation values are period-specific. Make them period-specific.
    cont_indices = indices - state_space.slices_by_periods[period].start
    continuation_values = state_space.get_continuation_values(period)[cont_indices]

    draws_shock = df[[f"shock_reward_{c}" for c in optim_paras["choices"]]].to_numpy()
    draws_wage = df[[f"meas_error_wage_{c}" for c in optim_paras["choices"]]].to_numpy()

    wages = state_space.get_attribute("wages")[indices]
    nonpecs = state_space.get_attribute("nonpecs")[indices]
    is_inadmissible = state_space.get_attribute("is_inadmissible")[indices]

    value_functions, flow_utilities = calculate_value_functions_and_flow_utilities(
        wages,
        nonpecs,
        continuation_values,
        draws_shock,
        optim_paras["delta"],
        is_inadmissible,
    )

    # We need to ensure that no individual chooses an inadmissible state. Thus, set
    # value functions to NaN. This cannot be done in
    # :func:`aggregate_keane_wolpin_utility` as the interpolation requires a mild
    # penalty.
    value_functions = np.where(is_inadmissible, np.nan, value_functions)

    choice = np.nanargmax(value_functions, axis=1)

    wages = wages * draws_shock * draws_wage
    wages[:, n_wages:] = np.nan
    wage = np.choose(choice, wages.T)

    # Store necessary information and information for debugging, etc..
    df["choice"] = choice
    df["wage"] = wage
    df["discount_rate"] = optim_paras["delta"]
    for i, choice in enumerate(optim_paras["choices"]):
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
    for column in all_data:
        if all_data[column].dtype == np.bool:
            all_data[column] = all_data[column].astype(np.uint8)

    # Calculate dot product of covariates and parameters.
    z = ()
    for level in level_dict:
        labels = level_dict[level].index
        x_beta = np.dot(all_data[labels], level_dict[level])

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

    This function takes an array of simulated outcomes for each period and stacks them
    together to one DataFrame.

    Parameters
    ----------
    data : list
        List of DataFrames which contains the simulated data with internal codes and
        labels.
    optim_paras : dict

    Returns
    -------
    simulated_df : pandas.DataFrame
        DataFrame with simulated data.

    """
    df = (
        pd.concat(data)
        .rename(columns=rename_labels_from_internal)
        .rename_axis(index=rename_labels_from_internal)
    )
    df = _convert_codes_to_original_labels(df, optim_paras)

    # We use the downcast to convert some variables to integers.
    df = df.apply(downcast_to_smallest_dtype)

    return df


def _random_choice(choices, probabilities, decimals=5):
    """Return elements of choices for a two-dimensional array of probabilities.

    It is assumed that probabilities are ordered (n_samples, n_choices).

    The function is taken from this `StackOverflow post
    <https://stackoverflow.com/questions/40474436>`_ as a workaround for
    :func:`numpy.random.choice` as it can only handle one-dimensional probabilities.

    Example
    -------
    Here is an example with non-zero probabilities.

    >>> n_samples = 100_000
    >>> choices = np.array([0, 1, 2])
    >>> p = np.array([0.15, 0.35, 0.5])
    >>> ps = np.tile(p, (n_samples, 1))
    >>> choices = _random_choice(choices, ps)
    >>> np.round(np.bincount(choices), decimals=-3) / n_samples
    array([0.15, 0.35, 0.5 ])

    Here is an example where one choice has probability zero.

    >>> p = np.array([0.4, 0, 0.6])
    >>> ps = np.tile(p, (n_samples, 1))
    >>> choices = _random_choice(choices, ps)
    >>> np.round(np.bincount(choices), decimals=-3) / n_samples
    array([0.4, 0. , 0.6])

    """
    cumulative_distribution = probabilities.cumsum(axis=1)
    # Probabilities often do not sum to one but 0.99999999999999999.
    cumulative_distribution[:, -1] = np.round(cumulative_distribution[:, -1], decimals)

    if not (cumulative_distribution[:, -1] == 1).all():
        raise ValueError("Probabilities do not sum to one.")

    u = np.random.rand(cumulative_distribution.shape[0], 1)

    # Note that :func:`np.argmax` returns the first index for multiple maximum values.
    indices = (u < cumulative_distribution).argmax(axis=1)

    if isinstance(choices, int):
        choices = np.arange(choices)
    elif isinstance(choices, (dict, list, np.ndarray, tuple)):
        choices = np.array(list(choices))
    else:
        raise TypeError(f"'choices' has invalid type {type(choices)}.")

    return choices[indices]


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


def _harmonize_simulation_arguments(method, df, n_sim_p, options):
    """Harmonize the arguments of the simulation."""
    if method == "n_step_ahead_with_sampling":
        pass
    else:
        if df is None:
            raise ValueError(f"Method '{method}' requires data.")

        options["simulation_agents"] = df.index.get_level_values("Identifier").nunique()

        if method == "one_step_ahead":
            n_sim_p = int(df.index.get_level_values("Period").max() + 1)

    n_sim_p = options["n_periods"] if n_sim_p is None else n_sim_p
    if options["n_periods"] < n_sim_p:
        options["n_periods"] = n_sim_p
        warnings.warn(
            f"The number of periods in the model, {options['n_periods']}, is lower than"
            f" the requested number of simulated periods, {n_sim_p}. Set model periods "
            "equal to simulated periods."
        )

    return n_sim_p, options


def _process_input_df_for_simulation(df, method, n_sim_periods, options, optim_paras):
    """Process the ``df`` provided by the user for the simulation."""
    if method == "n_step_ahead_with_sampling":
        ids = np.arange(options["simulation_agents"])
        index = pd.MultiIndex.from_product(
            (ids, range(n_sim_periods)), names=["identifier", "period"]
        )
        df = pd.DataFrame(index=index)

    elif method == "n_step_ahead_with_data":
        ids = np.arange(options["simulation_agents"])
        index = pd.MultiIndex.from_product(
            (ids, range(n_sim_periods)), names=["identifier", "period"]
        )
        df = (
            df.copy()
            .rename(columns=rename_labels_to_internal)
            .rename_axis(index=rename_labels_to_internal)
            .query("period == 0")
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
