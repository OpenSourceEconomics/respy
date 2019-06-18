import numpy as np
import pandas as pd
from numba import guvectorize

from respy.config import HUGE_FLOAT
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import _aggregate_keane_wolpin_utility
from respy.shared import _generate_column_labels_simulation
from respy.shared import create_base_draws
from respy.shared import get_conditional_probabilities
from respy.shared import transform_disturbances
from respy.solve import solve_with_backward_induction
from respy.state_space import StateSpace


def simulate(params, options):
    """Simulate a data set.

    This function provides the interface for the simulation of a data set.

    Parameters
    ----------
    params : pd.DataFrame or pd.Series
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
    state_space : class
        Class of state space.
    base_draws_sim : np.ndarray
        Array with shape (n_periods, n_agents_sim, n_choices).
    base_draws_wage : np.ndarray
        Array with shape (n_periods, n_agents_sim, n_choices).
    optim_paras : dict
        Parameters affected by optimization.
    options : dict
        Dictionary containing model options.

    Returns
    -------
    simulated_data : pd.DataFrame
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
            base_draws_sim[period], np.zeros(n_choices), optim_paras["shocks_cholesky"]
        )

    base_draws_wage_transformed = np.exp(base_draws_wage * optim_paras["meas_error"])

    # Create initial starting values for agents in simulation.
    container = ()
    for choice in options["choices_w_exp"]:
        container += (_get_random_initial_experience(choice, options),)
    container += (_get_random_lagged_choices(container[2], options),)
    container += (_get_random_types(container[2], optim_paras, options),)

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


def _sort_type_info(optim_paras):
    """We fix an order for the sampling of the types."""
    type_shares = optim_paras["type_shares"].reshape(-1, 2)
    order = np.argsort(type_shares[:, 0])
    type_info = {"shares": type_shares[order].flatten(), "order": order}

    return type_info


def _get_random_types(edu_start, optim_paras, options):
    """Get random types for simulated agents."""
    # We want to ensure that the order of types in the initialization file does not
    # matter for the simulated sample.
    type_info = _sort_type_info(optim_paras)

    np.random.seed(options["simulation_seed"])

    types = []
    for i in range(options["simulation_agents"]):
        probs = get_conditional_probabilities(
            type_info["shares"], np.array([edu_start[i]])
        )
        types += np.random.choice(type_info["order"], p=probs, size=1).tolist()

    # If we only have one individual, we need to ensure that types are a vector.
    types = np.array(types, ndmin=1)

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
    wages : np.ndarray
        Array with shape (n_choices,).
    nonpec : np.ndarray
        Array with shape (n_choices,).
    continuation_values : np.ndarray
        Array with shape (n_choices,)
    draws : np.ndarray
        Array with shape (n_draws, n_choices)
    delta : float
        Discount rate.
    is_inadmissible: np.ndarray
        Array with shape (n_choices,) containing indicator for whether the following
        state is inadmissible.

    Returns
    -------
    value_functions : np.ndarray
        Array with shape (n_choices, n_draws).
    flow_utilities : np.ndarray
        Array with shape (n_choices, n_draws)

    """
    n_draws, n_choices = draws.shape

    for i in range(n_draws):
        for j in range(n_choices):
            value_function, flow_utility = _aggregate_keane_wolpin_utility(
                wages[j],
                nonpec[j],
                continuation_values[j],
                draws[i, j],
                delta,
                is_inadmissible[j],
            )

            flow_utilities[j, i] = flow_utility
            value_functions[j, i] = value_function


def _process_simulated_data(data, options):
    labels, dtypes = _generate_column_labels_simulation(options)

    df = (
        pd.DataFrame(data=np.vstack(data), columns=labels)
        .astype(dtypes)
        .sort_values(["Identifier", "Period"])
        .reset_index(drop=True)
    )

    code_to_choice = {i: choice for i, choice in enumerate(options["choices"])}

    df.Choice = df.Choice.cat.set_categories(code_to_choice).cat.rename_categories(
        code_to_choice
    )
    df.Lagged_Choice = df.Lagged_Choice.cat.set_categories(
        code_to_choice
    ).cat.rename_categories(code_to_choice)

    return df
