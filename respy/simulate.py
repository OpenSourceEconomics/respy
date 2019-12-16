"""Everything related to the simulation of data with structural models."""
import functools
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax

from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import calculate_value_functions_and_flow_utilities
from respy.shared import create_base_covariates
from respy.shared import create_base_draws
from respy.shared import create_type_covariates
from respy.shared import generate_column_labels_simulation
from respy.shared import rename_labels
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

    df, n_simulation_periods, options = _process_simulation_arguments(
        method, df, n_simulation_periods, options
    )

    state_space = StateSpace(optim_paras, options)

    shape = (
        n_simulation_periods,
        options["simulation_agents"],
        len(optim_paras["choices"]),
    )
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
        state_space=state_space,
        options=options,
        df=df,
    )

    return simulate_function


def simulate(params, options, df, state_space, base_draws_sim, base_draws_wage):
    """Perform a simulation.

    This function performs one of three possible simulation exercises. Ordered from no
    data to a panel data on individuals, there is:

    1. *n-step-ahead simulation without data*: The first observation of an individual is
       sampled from the initial conditions, i.e., the distribution of observed variables
       or initial experiences, etc. in the first period. Then, the individuals are
       guided for $n$ periods by the decision rules from the solution of the model.

    2. *n-step-ahead simulation with first observations*: Instead of sampling
       individuals from the initial conditions, take the first observation of each
       individual in the data. Then, do as in 1..

    3. *one-step-ahead simulation*: Take the complete data and find for each observation
       the corresponding outcomes, e.g, choices and wages, using the decision rules from
       the model solution.

    Parameters
    ----------
    params : pandas.DataFrame or pandas.Series
        Contains parameters.
    options : dict
        Contains model options.
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
    base_draws_sim : np.ndarray
        Array with shape (n_periods, n_individuals, n_choices) to provide a unique set
        of shocks for each individual in each period.
    base_draws_wage : np.ndarray
        Array with shape (n_periods, n_individuals, n_choices) to provide a unique set
        of wage measurement errors for each individual in each period.

    Returns
    -------
    simulated_data : pandas.DataFrame
        DataFrame of simulated individuals.

    """
    optim_paras, options = process_params_and_options(params, options)

    # Solve the model.
    state_space.update_systematic_rewards(optim_paras)
    state_space = solve_with_backward_induction(state_space, optim_paras, options)

    # Adjust options and prepare data. If no data is passed or if only one observation
    # for each individual is passed, perform n-step-ahead simulation. Else perform
    # one-step-ahead simulation. Also, set a flag for the simulation method and adjust
    # the number of periods.
    if df is None:
        df = _sample_data_from_initial_conditions(optim_paras, options)

        is_n_step_ahead = True
        n_periods = options["n_periods"]
        n_individuals = options["simulation_agents"]

    else:
        df = _sample_unobserved_data_from_initial_conditions(df, optim_paras, options)

        is_n_step_ahead = (
            df.index.get_level_values("identifier").duplicated().sum() == 0
        )
        n_periods = (
            options["n_periods"]
            if is_n_step_ahead
            else int(df.index.get_level_values("period").max() + 1)
        )
        n_individuals = df.index.get_level_values("identifier").nunique()

    # Transform shocks
    n_wages = len(optim_paras["choices_w_wage"])
    base_draws_sim_transformed = transform_base_draws_with_cholesky_factor(
        base_draws_sim, optim_paras["shocks_cholesky"], n_wages
    )
    base_draws_wage_transformed = np.exp(base_draws_wage * optim_paras["meas_error"])

    state_space_columns = _create_state_space_columns(optim_paras)
    data = []

    for period in range(n_periods):

        current_states = df.query("period == @period")[state_space_columns].to_numpy(
            dtype=np.uint32
        )

        rows = _simulate_single_period(
            period,
            current_states,
            state_space,
            base_draws_sim_transformed,
            base_draws_wage_transformed,
            optim_paras,
        )

        data.append(rows)

        if is_n_step_ahead:
            choice = rows[:, 1].astype(np.uint8)
            df = apply_law_of_motion(choice, current_states, period, optim_paras)

    if is_n_step_ahead:
        identifier = np.tile(np.arange(n_individuals), n_periods)
    else:
        identifier = df.index.get_level_values("identifier")

    simulated_data = _process_simulated_data(identifier, data, optim_paras)

    return simulated_data


def _sample_data_from_initial_conditions(optim_paras, options):
    """Sample initial observations from initial conditions."""
    index = pd.MultiIndex.from_product(
        (np.arange(options["simulation_agents"]), [0]), names=["identifier", "period"]
    )
    states_df = pd.DataFrame(index=index)
    # Create observables for simulation and store them in an extra container that
    # will be added to the state space container later
    for observable in optim_paras["observables"]:
        level_dict = optim_paras["observables"][observable]
        states_df[observable] = _get_random_characteristic(
            states_df, options, level_dict, use_keys=False
        )

    # Create initial experiences, lagged choices and types for agents in simulation.
    for choice in optim_paras["choices_w_exp"]:
        level_dict = optim_paras["choices"][choice]["start"]
        states_df[f"exp_{choice}"] = _get_random_characteristic(
            states_df, options, level_dict, use_keys=True
        )
    for lag in reversed(range(1, optim_paras["n_lagged_choices"] + 1)):
        level_dict = optim_paras[f"lagged_choice_{lag}"]
        states_df[f"lagged_choice_{lag}"] = _get_random_characteristic(
            states_df, options, level_dict, use_keys=False
        )

    df = _get_random_types(states_df, optim_paras, options)

    return df


def _sample_unobserved_data_from_initial_conditions(df, optim_paras, options):
    """Add unobserved information to input data."""
    df = df.copy()

    if df.index.names != ["Identifier", "Period"]:
        raise ValueError(
            "Input data needs to have a pd.MultiIndex with Identifier and Period."
        )

    df = df.rename(columns=rename_labels).rename_axis(index=rename_labels).sort_index()

    # Assign a type to each individual which is unobserved by the researcher.
    first_period_df = df.query("period == 0").copy()
    first_period_df = _get_random_types(first_period_df, optim_paras, options)

    df = df.join(first_period_df["type"])

    return df


def _simulate_single_period(
    period,
    current_states,
    state_space,
    base_draws_sim_transformed,
    base_draws_wage_transformed,
    optim_paras,
):
    """Simulate individuals in a single period.

    This function takes a set of states and simulates wages, choices and other
    information.

    """
    n_wages = len(optim_paras["choices_w_wage"])
    n_individuals = current_states.shape[0]

    # Get indices which connect states in the state space and simulated agents.
    indices = state_space.indexer[period][
        tuple(current_states[:, i] for i in range(current_states.shape[1]))
    ]

    # Get continuation values. Indices work on the complete state space whereas
    # continuation values are period-specific. Make them period-specific.
    continuation_values = state_space.get_continuation_values(period)
    cont_indices = indices - state_space.slices_by_periods[period].start

    # Select relevant subset of random draws.
    draws_shock = base_draws_sim_transformed[period][:n_individuals]
    draws_wage = base_draws_wage_transformed[period][:n_individuals]

    # Get total values and ex post rewards.
    value_functions, flow_utilities = calculate_value_functions_and_flow_utilities(
        state_space.wages[indices],
        state_space.nonpec[indices],
        continuation_values[cont_indices],
        draws_shock,
        optim_paras["delta"],
        state_space.is_inadmissible[indices],
    )

    # We need to ensure that no individual chooses an inadmissible state. Thus, set
    # value functions to NaN. This cannot be done in
    # :func:`aggregate_keane_wolpin_utility` as the interpolation requires a mild
    # penalty.
    value_functions = np.where(
        state_space.is_inadmissible[indices], np.nan, value_functions
    )

    choice = np.nanargmax(value_functions, axis=1)

    wages = state_space.wages[indices] * draws_shock * draws_wage
    wages[:, n_wages:] = np.nan
    wage = np.choose(choice, wages.T)

    # Record data of all agents in one period.
    rows = np.column_stack(
        (
            np.full(n_individuals, period),
            choice,
            wage,
            # Write relevant state space for period to data frame. However, the
            # individual's type is not part of the observed dataset. This is
            # included in the simulated dataset.
            current_states,
            # As we are working with a simulated dataset, we can also output
            # additional information that is not available in an observed dataset.
            # The discount rate is included as this allows to construct the EMAX
            # with the information provided in the simulation output.
            state_space.nonpec[indices],
            state_space.wages[indices, :n_wages],
            flow_utilities,
            value_functions,
            draws_shock,
            np.full(n_individuals, optim_paras["delta"]),
        )
    )

    return rows


def _get_random_types(states_df, optim_paras, options):
    """Get random types for simulated agents."""
    if optim_paras["n_types"] == 1:
        states_df["type"] = np.zeros(options["simulation_agents"], dtype=np.uint8)
    else:
        type_covariates = create_type_covariates(states_df, optim_paras, options)
        np.random.seed(next(options["simulation_seed_iteration"]))

        z = np.dot(type_covariates, optim_paras["type_prob"].T)
        probs = softmax(z, axis=1)
        states_df["type"] = _random_choice(optim_paras["n_types"], probs)

    return states_df


def _get_random_characteristic(states_df, options, level_dict, use_keys):
    """Sample characteristic of individuals.

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

    """
    covariates_df = create_base_covariates(
        states_df, options["covariates"], raise_errors=False
    )

    all_data = pd.concat([covariates_df, states_df], axis="columns", sort=False)

    z = ()

    for level in level_dict:
        labels = level_dict[level].index
        x_beta = np.dot(all_data[labels], level_dict[level])

        z += (x_beta,)

    probabilities = softmax(np.column_stack(z), axis=1)

    np.random.seed(next(options["simulation_seed_iteration"]))

    choices = level_dict if use_keys else len(level_dict)
    characteristic = _random_choice(choices, probabilities)

    return characteristic


def _convert_codes_to_original_labels(df, optim_paras):
    code_to_choice = dict(enumerate(optim_paras["choices"]))

    df.Choice = df.Choice.cat.set_categories(code_to_choice).cat.rename_categories(
        code_to_choice
    )
    for i in range(1, optim_paras["n_lagged_choices"] + 1):
        df[f"Lagged_Choice_{i}"] = (
            df[f"Lagged_Choice_{i}"]
            .cat.set_categories(code_to_choice)
            .cat.rename_categories(code_to_choice)
        )

    for observable in optim_paras["observables"]:
        code_to_obs = dict(enumerate(optim_paras["observables"][observable]))
        df[f"{observable.title()}"] = df[f"{observable.title()}"].replace(code_to_obs)

    return df


def _process_simulated_data(identifier, data, optim_paras):
    labels, dtypes = generate_column_labels_simulation(optim_paras)

    df = (
        pd.DataFrame(
            data=np.column_stack((identifier, np.row_stack(data))), columns=labels
        )
        .astype(dtypes)
        .sort_values(["Identifier", "Period"])
        .set_index(["Identifier", "Period"], drop=True)
    )

    df = _convert_codes_to_original_labels(df, optim_paras)

    return df


def _random_choice(choices, probabilities):
    """Return elements of choices for a two-dimensional array of probabilities.

    It is assumed that probabilities are ordered (n_samples, n_choices).

    The function is taken from this `StackOverflow post
    <https://stackoverflow.com/questions/40474436>`_ as a workaround for
    :func:`np.random.choice` as it can only handle one-dimensional probabilities.

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
    cumulative_distribution[:, -1] = np.round(cumulative_distribution[:, -1], 5)

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


def apply_law_of_motion(choice, states, period, optim_paras):
    """Apply the law of motion to get the states in the next period.

    This function changes experiences and previous choices according to the choice in
    the current period, to get the states of the next period.

    Parameters
    ----------
    choice : np.ndarray
        Array with shape (n_individuals,) containing the current choice.
    states : np.ndarray
        Array with shape (n_individuals, n_state_space_dim) containing the state of each
        individual.

    """
    n_individuals = states.shape[0]
    n_choices_w_exp = len(optim_paras["choices_w_exp"])
    n_lagged_choices = optim_paras["n_lagged_choices"]

    # Update work experiences.
    states[np.arange(n_individuals), choice] = np.where(
        choice < n_choices_w_exp,
        states[np.arange(n_individuals), choice] + 1,
        states[np.arange(n_individuals), choice],
    )

    # Update lagged choices by shifting all lags by one and inserting choice in
    # the first position.
    if n_lagged_choices:
        states[:, n_choices_w_exp + 1 : n_choices_w_exp + n_lagged_choices] = states[
            :, n_choices_w_exp : n_choices_w_exp + n_lagged_choices - 1
        ]
        states[:, n_choices_w_exp] = choice

    state_space_columns = _create_state_space_columns(optim_paras)
    df = pd.DataFrame(states, columns=state_space_columns)
    df.insert(0, "period", period + 1)

    return df


def _create_state_space_columns(optim_paras):
    """Create a list of state space dimensions in order excluding `"period"`."""
    return (
        [f"exp_{choice}" for choice in optim_paras["choices_w_exp"]]
        + [f"lagged_choice_{i}" for i in range(1, optim_paras["n_lagged_choices"] + 1)]
        + list(optim_paras["observables"])
        + ["type"]
    )


def _process_simulation_arguments(method, df, n_simulation_periods, options):
    if n_simulation_periods is None:
        n_simulation_periods = options["n_periods"]

    if method == "n_step_ahead_with_sampling":
        if df is not None:
            df = None
            warnings.warn("Passed df is ignored.")

    elif method == "n_step_ahead_with_data":
        df = df.query("Period == 0")
        options["simulation_agents"] = df.index.get_level_values("Identifier").nunique()

    elif method == "one_step_ahead":
        if df is None:
            raise ValueError(f"Method '{method}' requires data.")
        options["simulation_agents"] = df.index.get_level_values("Identifier").nunique()

    else:
        raise NotImplementedError(f"Simulation method '{method}' is not implemented.")

    if options["n_periods"] < n_simulation_periods:
        options["n_periods"] = n_simulation_periods
        warnings.warn(
            f"The number of periods in the model, {options['n_periods']}, is lower than"
            f" the requested number of simulated periods, {n_simulation_periods}. Set "
            "model periods equal to simulated periods."
        )

    return df, n_simulation_periods, options
