import numpy as np
import pandas as pd
from numba import guvectorize

from respy.config import HUGE_FLOAT
from respy.likelihood import create_type_covariates
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import aggregate_keane_wolpin_utility
from respy.shared import create_base_draws
from respy.shared import generate_column_labels_simulation
from respy.shared import predict_multinomial_logit
from respy.shared import transform_disturbances
from respy.solve import solve_with_backward_induction
from respy.state_space import StateSpace


def simulate(params, options):
    """Simulate a data set.

    This function provides the interface for the simulation of a data set.

    Parameters
    ----------
    params : pandas.DataFrame or pandas.Series
        DataFrame or Series containing parameters.
    options : dict
        Dictionary containing model options.

    """
    params, optim_paras, options = process_params_and_options(params, options)

    state_space = StateSpace(params, options)
    state_space = solve_with_backward_induction(state_space, optim_paras, options)

    base_draws_sim = create_base_draws(
        (options["n_periods"], options["simulation_agents"], len(options["choices"])),
        options["simulation_seed"],
    )

    # ``seed + 1`` ensures that draws for wages are different than for simulation.
    base_draws_wage = create_base_draws(
        base_draws_sim.shape, seed=options["simulation_seed"] + 1
    )

    simulated_data = simulate_data(
        state_space, base_draws_sim, base_draws_wage, optim_paras, options
    )

    return state_space, simulated_data


def simulate_data(state_space, base_draws_sim, base_draws_wage, optim_paras, options):
    """Simulate a data set.

    At the beginning, agents are initialized with zero experience in occupations and
    random values for years of education, lagged choices and types. Then, each simulated
    agent in each period is paired with its corresponding state in the state space. We
    recalculate utilities for each choice as the agents experience different shocks in
    the simulation. In the end, observed and unobserved information is recorded in the
    simulated dataset.

    Parameters
    ----------
    state_space : :class:`~respy.state_space.StateSpace`
        Class of state space.
    base_draws_sim : numpy.ndarray
        Array with shape (n_periods, n_agents_sim, n_choices).
    base_draws_wage : numpy.ndarray
        Array with shape (n_periods, n_agents_sim, n_choices).
    optim_paras : dict
        Parameters affected by optimization.
    options : dict
        Dictionary containing model options.

    Returns
    -------
    simulated_data : pandas.DataFrame
        Dataset of simulated agents.

    """
    n_choices = len(options["choices"])
    n_periods = options["n_periods"]
    n_wages = len(options["choices_w_wage"])

    # Standard deviates transformed to the distributions relevant for the agents actual
    # decision making as traversing the tree.
    base_draws_sim_transformed = np.full(
        (n_periods, options["simulation_agents"], n_choices), np.nan
    )

    for period in range(n_periods):
        base_draws_sim_transformed[period] = transform_disturbances(
            base_draws_sim[period],
            np.zeros(n_choices),
            optim_paras["shocks_cholesky"],
            n_wages,
        )

    base_draws_wage_transformed = np.exp(base_draws_wage * optim_paras["meas_error"])

    # Create initial starting values for agents in simulation.
    container = ()
    for choice in options["choices_w_exp"]:
        container += (_get_random_initial_experience(choice, options),)

    edu_idx = list(options["choices_w_exp"]).index("edu")
    container += (_get_random_lagged_choices(container[edu_idx], options),)
    states_wo_types = pd.DataFrame(
        np.column_stack(container),
        columns=[f"exp_{i}" for i in options["choices_w_exp"]] + ["lagged_choice"],
    ).assign(period=0)
    container += (_get_random_types(states_wo_types, optim_paras, options),)

    # Create a matrix of initial states of simulated agents.
    current_states = np.column_stack(container).astype(np.uint8)

    data = []

    for period in range(n_periods):

        # Get indices which connect states in the state space and simulated agents.
        ks = state_space.indexer[
            (np.full(options["simulation_agents"], period),)
            + tuple(current_states[:, i] for i in range(current_states.shape[1]))
        ]

        # Select relevant subset of random draws.
        draws_shock = base_draws_sim_transformed[period]
        draws_wage = base_draws_wage_transformed[period]

        # Get total values and ex post rewards.
        value_functions, flow_utilities = calculate_value_functions_and_flow_utilities(
            state_space.wages[ks],
            state_space.nonpec[ks],
            state_space.continuation_values[ks],
            draws_shock.reshape(-1, 1, n_choices),
            optim_paras["delta"],
            state_space.is_inadmissible[ks],
        )
        value_functions = value_functions.reshape(-1, n_choices)
        flow_utilities = flow_utilities.reshape(-1, n_choices)

        # We need to ensure that no individual chooses an inadmissible state. This
        # cannot be done directly in the calculate_value_functions function as the
        # penalty otherwise dominates the interpolation equation. The parameter
        # INADMISSIBILITY_PENALTY is a compromise. It is only relevant in very
        # constructed cases.
        value_functions = np.where(
            state_space.is_inadmissible[ks], -HUGE_FLOAT, value_functions
        )

        # Determine optimal choice.
        choice = np.argmax(value_functions, axis=1)

        wages = state_space.wages[ks] * draws_shock * draws_wage
        wages[:, n_wages:] = np.nan
        wage = np.choose(choice, wages.T)

        # Record data of all agents in one period.
        rows = np.column_stack(
            (
                np.arange(options["simulation_agents"]),
                np.full(options["simulation_agents"], period),
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
                state_space.nonpec[ks],
                state_space.wages[ks, :n_wages],
                flow_utilities,
                value_functions,
                draws_shock,
                np.full(options["simulation_agents"], optim_paras["delta"][0]),
            )
        )
        data.append(rows)

        # Update work experiences.
        current_states[np.arange(options["simulation_agents"]), choice] = np.where(
            choice <= len(options["choices_w_exp"]),
            current_states[np.arange(options["simulation_agents"]), choice] + 1,
            current_states[np.arange(options["simulation_agents"]), choice],
        )
        # Update lagged choices.
        current_states[:, -2] = choice

    simulated_data = _process_simulated_data(data, options)

    return simulated_data


def _get_random_types(states, optim_paras, options):
    """Get random types for simulated agents."""
    if options["n_types"] == 1:
        types = np.zeros(options["simulation_agents"])
    else:
        type_covariates = create_type_covariates(states, options)

        np.random.seed(options["simulation_seed"])

        probs = predict_multinomial_logit(optim_paras["type_prob"], type_covariates)
        types = _random_choice(options["n_types"], probs)

    return types


def _get_random_initial_experience(choice, options):
    """Get random, initial levels of schooling for simulated agents."""
    np.random.seed(options["simulation_seed"])

    initial_experience = np.random.choice(
        options["choices"][choice]["start"],
        p=options["choices"][choice]["share"],
        size=options["simulation_agents"],
    )

    return initial_experience


def _get_random_lagged_choices(edu_start, options):
    """Get random, initial levels of lagged choices for simulated agents."""
    np.random.seed(options["simulation_seed"])

    choices = [
        list(options["choices"]).index("edu"),
        list(options["choices"]).index("home"),
    ]

    lagged_start = []
    for i in range(options["simulation_agents"]):
        idx = np.where(options["choices"]["edu"]["start"] == edu_start[i])[0][0]
        probs = (
            options["choices"]["edu"]["lagged"][idx],
            1 - options["choices"]["edu"]["lagged"][idx],
        )
        lagged_start += np.random.choice(choices, p=probs, size=1).tolist()

    # If we only have one individual, we need to ensure that activities are a vector.
    lagged_start = np.array(lagged_start, ndmin=1)

    return lagged_start


@guvectorize(
    [
        "f4[:], f4[:], f4[:], f4[:, :], f4, b1[:], f4[:, :], f4[:, :]",
        "f8[:], f8[:], f8[:], f8[:, :], f8, b1[:], f8[:, :], f8[:, :]",
    ],
    "(n_choices), (n_choices), (n_choices), (n_draws, n_choices), (), (n_choices) "
    "-> (n_choices, n_draws), (n_choices, n_draws)",
    nopython=True,
    target="cpu",
)
def calculate_value_functions_and_flow_utilities(
    wages,
    nonpec,
    continuation_values,
    draws,
    delta,
    is_inadmissible,
    value_functions,
    flow_utilities,
):
    """Calculate the choice-specific value functions and flow utilities.

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
        Array with shape (n_choices,) containing indicator for whether the following
        state is inadmissible.

    Returns
    -------
    value_functions : numpy.ndarray
        Array with shape (n_choices, n_draws).
    flow_utilities : numpy.ndarray
        Array with shape (n_choices, n_draws).

    """
    n_draws, n_choices = draws.shape

    for i in range(n_draws):
        for j in range(n_choices):
            value_function, flow_utility = aggregate_keane_wolpin_utility(
                wages[j],
                nonpec[j],
                continuation_values[j],
                draws[i, j],
                delta,
                is_inadmissible[j],
            )

            flow_utilities[j, i] = flow_utility
            value_functions[j, i] = value_function


def _convert_choice_variables_from_codes_to_categorical(df, options):
    code_to_choice = {i: choice for i, choice in enumerate(options["choices"])}

    df.Choice = df.Choice.cat.set_categories(code_to_choice).cat.rename_categories(
        code_to_choice
    )
    df.Lagged_Choice = df.Lagged_Choice.cat.set_categories(
        code_to_choice
    ).cat.rename_categories(code_to_choice)

    return df


def _process_simulated_data(data, options):
    labels, dtypes = generate_column_labels_simulation(options)

    df = (
        pd.DataFrame(data=np.vstack(data), columns=labels)
        .astype(dtypes)
        .sort_values(["Identifier", "Period"])
        .reset_index(drop=True)
    )

    df = _convert_choice_variables_from_codes_to_categorical(df, options)

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
    cumulative_distribution[:, -1] = np.round(cumulative_distribution[:, -1], 15)

    if not (cumulative_distribution[:, -1] == 1).all():
        raise ValueError("Probabilities do not sum to one.")

    u = np.random.rand(cumulative_distribution.shape[0], 1)

    # Note that :func:`np.argmax` returns the first index for multiple maximum values.
    indices = (u < cumulative_distribution).argmax(axis=1)

    if isinstance(choices, int):
        choices = np.arange(choices)

    return choices[indices]
