"""Everything related to the simulation of data with structural models."""
import functools

import numpy as np
import pandas as pd
from scipy.special import softmax

from respy.config import MIN_FLOAT
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import calculate_value_functions_and_flow_utilities
from respy.shared import create_base_covariates
from respy.shared import create_base_draws
from respy.shared import create_type_covariates
from respy.shared import generate_column_labels_simulation
from respy.shared import transform_base_draws_with_cholesky_factor
from respy.solve import solve_with_backward_induction
from respy.state_space import StateSpace


def get_simulate_func(params, options):
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

    Returns
    -------
    simulate_function : :func:`simulate`
        Simulation function where all arguments except the parameter vector are set.

    """
    optim_paras, options = process_params_and_options(params, options)

    state_space = StateSpace(optim_paras, options)

    shape = (
        options["n_periods"],
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
    )

    return simulate_function


def simulate(params, base_draws_sim, base_draws_wage, state_space, options):
    """Simulate a data set.

    This function provides the interface for the simulation of a data set.

    Parameters
    ----------
    params : pandas.DataFrame or pandas.Series
        Contains parameters.
    base_draws_sim : np.ndarray
        Array with shape (n_periods, n_individuals, n_choices) to provide a unique set
        of shocks for each individual in each period.
    base_draws_wage : np.ndarray
        Array with shape (n_periods, n_individuals, n_choices) to provide a unique set
        of wage measurement errors for each individual in each period.
    state_space : :class:`~respy.state_space.StateSpace`
        State space of the model.
    options : dict
        Dictionary containing model options.

    Returns
    -------
    simulated_data : pandas.DataFrame
        DataFrame of simulated individuals.

    """
    optim_paras, options = process_params_and_options(params, options)

    state_space.update_systematic_rewards(optim_paras)

    state_space = solve_with_backward_induction(state_space, optim_paras, options)

    simulated_data = _simulate_data(
        state_space, base_draws_sim, base_draws_wage, optim_paras, options
    )

    return simulated_data


def _simulate_data(state_space, base_draws_sim, base_draws_wage, optim_paras, options):
    """Simulate a data set.

    At the beginning, individuals are initialized with zero experience in occupations
    and random values for years of education, lagged choices and types. Then, each
    simulated agent in each period is paired with its corresponding state in the state
    space. We recalculate utilities for each choice as the individuals experience
    different shocks in the simulation. In the end, observed and unobserved information
    is recorded in a DataFrame.

    Returns
    -------
    simulated_data : pandas.DataFrame
        Dataset of simulated individuals.

    """
    n_choices = len(optim_paras["choices"])
    n_periods = optim_paras["n_periods"]
    n_wages = len(optim_paras["choices_w_wage"])
    n_choices_w_exp = len(optim_paras["choices_w_exp"])
    n_lagged_choices = optim_paras["n_lagged_choices"]
    n_individuals = options["simulation_agents"]

    # Standard deviates transformed to the distributions relevant for the agents actual
    # decision making as traversing the tree.
    base_draws_sim_transformed = np.full((n_periods, n_individuals, n_choices), np.nan)

    for period in range(n_periods):
        base_draws_sim_transformed[period] = transform_base_draws_with_cholesky_factor(
            base_draws_sim[period],
            np.zeros(n_choices),
            optim_paras["shocks_cholesky"],
            n_wages,
        )

    base_draws_wage_transformed = np.exp(base_draws_wage * optim_paras["meas_error"])

    # Create a DataFrame to collect states.
    states_df = pd.DataFrame({"identifier": np.arange(n_individuals)}).assign(period=0)

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
    for lag in reversed(range(1, n_lagged_choices + 1)):
        level_dict = optim_paras[f"lagged_choice_{lag}"]
        states_df[f"lagged_choice_{lag}"] = _get_random_characteristic(
            states_df, options, level_dict, use_keys=False
        )

    states_df = _get_random_types(states_df, optim_paras, options)

    # Create a matrix of initial states of simulated agents.
    state_space_cols = (
        [f"exp_{choice}" for choice in optim_paras["choices_w_exp"]]
        + [f"lagged_choice_{lag}" for lag in reversed(range(1, n_lagged_choices + 1))]
        + list(optim_paras["observables"])
        + ["type"]
    )
    current_states = states_df[state_space_cols].to_numpy(dtype=np.uint8)

    data = []

    for period in range(n_periods):

        # Get indices which connect states in the state space and simulated agents.
        indices = state_space.indexer[period][
            tuple(current_states[:, i] for i in range(current_states.shape[1]))
        ]

        # Get continuation values. Indices work on the complete state space whereas
        # continuation values are period-specific. Make them period-specific.
        continuation_values = state_space.get_continuation_values(period)
        cont_indices = indices - state_space.slices_by_periods[period].start

        # Select relevant subset of random draws.
        draws_shock = base_draws_sim_transformed[period]
        draws_wage = base_draws_wage_transformed[period]

        # Get total values and ex post rewards.
        value_functions, flow_utilities = calculate_value_functions_and_flow_utilities(
            state_space.wages[indices],
            state_space.nonpec[indices],
            continuation_values[cont_indices],
            draws_shock,
            optim_paras["delta"],
            state_space.is_inadmissible[indices],
        )

        # We need to ensure that no individual chooses an inadmissible state. This
        # cannot be done directly in the calculate_value_functions function as the
        # penalty otherwise dominates the interpolation equation. The parameter
        # INADMISSIBILITY_PENALTY is a compromise. It is only relevant in very
        # constructed cases.
        value_functions = np.where(
            state_space.is_inadmissible[indices], MIN_FLOAT, value_functions
        )

        # Determine optimal choice.
        choice = np.argmax(value_functions, axis=1)

        wages = state_space.wages[indices] * draws_shock * draws_wage
        wages[:, n_wages:] = np.nan
        wage = np.choose(choice, wages.T)

        # Record data of all agents in one period.
        rows = np.column_stack(
            (
                np.arange(n_individuals),
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
        data.append(rows)

        # Update work experiences.
        current_states[np.arange(n_individuals), choice] = np.where(
            choice < n_choices_w_exp,
            current_states[np.arange(n_individuals), choice] + 1,
            current_states[np.arange(n_individuals), choice],
        )

        # Update lagged choices by shifting all lags by one and inserting choice in the
        # first position.
        if n_lagged_choices:
            current_states[
                :, n_choices_w_exp + 1 : n_choices_w_exp + n_lagged_choices
            ] = current_states[
                :, n_choices_w_exp : n_choices_w_exp + n_lagged_choices - 1
            ]
            current_states[:, n_choices_w_exp] = choice

    simulated_data = _process_simulated_data(data, optim_paras)

    return simulated_data


def _get_random_types(states_df, optim_paras, options):
    """Get random types for simulated agents."""
    if optim_paras["n_types"] == 1:
        states_df["type"] = np.zeros(options["simulation_agents"], dtype=np.uint8)
    else:
        type_covariates = create_type_covariates(states_df, optim_paras, options)
        np.random.seed(next(options["simulation_seed_iteration"]))

        z = np.dot(type_covariates, optim_paras["type_prob"].T)
        probs = softmax(z, axis=1)
        states_df["type"] = pd.Series(_random_choice(optim_paras["n_types"], probs))

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


def _process_simulated_data(data, optim_paras):
    labels, dtypes = generate_column_labels_simulation(optim_paras)

    df = (
        pd.DataFrame(data=np.vstack(data), columns=labels)
        .astype(dtypes)
        .sort_values(["Identifier", "Period"])
        .reset_index(drop=True)
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
    >>> assert np.bincount(choices)[1] == 0

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
