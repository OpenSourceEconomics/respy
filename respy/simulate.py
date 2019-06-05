import numpy as np
import pandas as pd
from numba import guvectorize

from respy.config import DATA_FORMATS_SIM
from respy.config import DATA_LABELS_SIM
from respy.config import HUGE_FLOAT
from respy.config import INADMISSIBILITY_PENALTY
from respy.pre_processing.model_processing import process_options
from respy.pre_processing.model_processing import process_params
from respy.shared import create_base_draws
from respy.shared import get_conditional_probabilities
from respy.shared import transform_disturbances
from respy.solve import solve_with_backward_induction
from respy.solve import StateSpace


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
    params, optim_paras = process_params(params)
    options = process_options(options)

    state_space = StateSpace(params, options)
    state_space = solve_with_backward_induction(
        state_space, options["interpolation_points"], optim_paras
    )

    # Draw draws for the simulation.
    base_draws_sims = create_base_draws(
        (options["num_periods"], options["simulation_agents"], 4),
        options["simulation_seed"],
    )

    simulated_data = simulate_data(state_space, base_draws_sims, optim_paras, options)

    return state_space, simulated_data


def simulate_data(state_space, base_draws_sims, optim_paras, options):
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
    base_draws_sims : np.ndarray
        Array with shape (num_periods, num_agents_sim, num_choices)
    optim_paras : dict
        Parameters affected by optimization.
    options : dict
        Dictionary containing model options.

    Returns
    -------
    simulated_data : pd.DataFrame
        Dataset of simulated agents.

    """
    # Standard deviates transformed to the distributions relevant for the agents actual
    # decision making as traversing the tree.
    base_draws_sims_transformed = np.full(
        (state_space.num_periods, options["simulation_agents"], 4), np.nan
    )

    for period in range(state_space.num_periods):
        base_draws_sims_transformed[period] = transform_disturbances(
            base_draws_sims[period], np.zeros(4), optim_paras["shocks_cholesky"]
        )

    # Create initial starting values for agents in simulation.
    initial_education = _get_random_edu_start(options)
    initial_types = _get_random_types(initial_education, optim_paras, options)
    initial_lagged_choices = _get_random_lagged_choices(initial_education, options)

    # Create a matrix of initial states of simulated agents. OCCUPATION A and OCCUPATION
    # B are set to zero.
    current_states = np.column_stack(
        (
            np.zeros((options["simulation_agents"], 2)),
            initial_education,
            initial_lagged_choices,
            initial_types,
        )
    ).astype(np.uint8)

    data = []

    for period in range(state_space.num_periods):

        # Get indices which connect states in the state space and simulated agents.
        ks = state_space.indexer[
            np.full(options["simulation_agents"], period),
            current_states[:, 0],
            current_states[:, 1],
            current_states[:, 2],
            current_states[:, 3] - 1,
            current_states[:, 4],
        ]

        # Select relevant subset of random draws.
        draws = base_draws_sims_transformed[period]

        # Get total values and ex post rewards.
        total_values, rewards_ex_post = get_continuation_value_and_ex_post_rewards(
            state_space.wages[ks],
            state_space.nonpec[ks],
            state_space.emaxs[ks],
            draws.reshape(-1, 1, 4),
            optim_paras["delta"],
            state_space.states[ks, 3] >= state_space.edu_max,
        )
        total_values = total_values.reshape(-1, 4)
        rewards_ex_post = rewards_ex_post.reshape(-1, 4)

        # We need to ensure that no individual chooses an inadmissible state. This
        # cannot be done directly in the get_continuation_value function as the penalty
        # otherwise dominates the interpolation equation. The parameter
        # INADMISSIBILITY_PENALTY is a compromise. It is only relevant in very
        # constructed cases.
        total_values[:, 2] = np.where(
            current_states[:, 2] >= state_space.edu_max, -HUGE_FLOAT, total_values[:, 2]
        )

        # Determine optimal choice.
        max_idx = np.argmax(total_values, axis=1)

        # Record wages. Expand matrix with NaNs for choice 2 and 3 for easier indexing.
        wages = (
            np.column_stack(
                (
                    state_space.wages[ks],
                    np.full((options["simulation_agents"], 2), np.nan),
                )
            )
            * draws
        )
        rewards_systematic = state_space.nonpec[ks]
        rewards_systematic[:, :2] += state_space.wages[ks]
        # Do not swap np.arange with : (https://stackoverflow.com/a/46425896/7523785)!
        wage = wages[np.arange(options["simulation_agents"]), max_idx]

        # Record data of all agents in one period.
        rows = np.column_stack(
            (
                np.arange(options["simulation_agents"]),
                np.full(options["simulation_agents"], period),
                max_idx + 1,
                wage,
                # Write relevant state space for period to data frame. However, the
                # individual's type is not part of the observed dataset. This is
                # included in the simulated dataset.
                current_states,
                # As we are working with a simulated dataset, we can also output
                # additional information that is not available in an observed dataset.
                # The discount rate is included as this allows to construct the EMAX
                # with the information provided in the simulation output.
                total_values,
                rewards_systematic,
                draws,
                np.full(options["simulation_agents"], optim_paras["delta"][0]),
                rewards_ex_post,
            )
        )
        data.append(rows)

        # Update work experiences or education and lagged choice for the next period.
        current_states[np.arange(options["simulation_agents"]), max_idx] = np.where(
            max_idx <= 2,
            current_states[np.arange(options["simulation_agents"]), max_idx] + 1,
            current_states[np.arange(options["simulation_agents"]), max_idx],
        )
        current_states[:, 3] = max_idx + 1

    simulated_data = (
        pd.DataFrame(data=np.vstack(data), columns=DATA_LABELS_SIM)
        .astype(DATA_FORMATS_SIM)
        .sort_values(["Identifier", "Period"])
    )

    return simulated_data


def _sort_type_info(optim_paras):
    """We fix an order for the sampling of the types."""
    type_info = {"order": np.argsort(optim_paras["type_shares"].tolist()[0::2])}

    # We simply fix the order by the size of the intercepts.

    # We need to reorder the coefficients determining the type probabilities
    # accordingly.
    type_shares = []
    for i in range(optim_paras["num_types"]):
        lower, upper = i * 2, (i + 1) * 2
        type_shares += [optim_paras["type_shares"][lower:upper].tolist()]
    type_info["shares"] = np.array(
        [type_shares[i] for i in type_info["order"]]
    ).flatten()

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


def _get_random_edu_start(options):
    """Get random, initial levels of schooling for simulated agents."""
    np.random.seed(options["simulation_seed"])

    # As we do not want to be too strict at the user-level the sum of edu_spec might
    # be slightly larger than one. This needs to be corrected here.
    probs = options["education_share"] / np.sum(options["education_share"])
    edu_start = np.random.choice(
        options["education_start"], p=probs, size=options["simulation_agents"]
    )

    # If we only have one individual, we need to ensure that types are a vector.
    edu_start = np.array(edu_start, ndmin=1)

    return edu_start


def _get_random_lagged_choices(edu_start, options):
    """Get random, initial levels of lagged choices for simulated agents."""
    np.random.seed(options["simulation_seed"])

    lagged_start = []
    for i in range(options["simulation_agents"]):
        idx = options["education_start"].index(edu_start[i])
        probs = options["education_lagged"][idx], 1 - options["education_lagged"][idx]
        lagged_start += np.random.choice([3, 4], p=probs, size=1).tolist()

    # If we only have one individual, we need to ensure that activities are a vector.
    lagged_start = np.array(lagged_start, ndmin=1)

    return lagged_start


@guvectorize(
    [
        "f4[:], f4[:], f4[:], f4[:, :], f4, b1, f4[:, :], f4[:, :]",
        "f8[:], f8[:], f8[:], f8[:, :], f8, b1, f8[:, :], f8[:, :]",
    ],
    "(m), (n), (n), (p, n), (), () -> (n, p), (n, p)",
    nopython=True,
    target="cpu",
)
def get_continuation_value_and_ex_post_rewards(
    wages, nonpec, emaxs, draws, delta, max_education, continuation_value, rew_ex_post
):
    """Calculate the continuation value and ex-post rewards.

    This function is a generalized ufunc which is flexible in the number of individuals
    and draws.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (2,).
    nonpec : np.ndarray
        Array with shape (4,).
    emaxs : np.ndarray
        Array with shape (4,)
    draws : np.ndarray
        Array with shape (num_draws, 4)
    delta : float
        Discount rate.
    max_education: bool
        Indicator for whether the state has reached maximum education.

    Returns
    -------
    continuation_value : np.ndarray
        Array with shape (4, num_draws).
    rew_ex_post : np.ndarray
        Array with shape (4, num_draws)

    Examples
    --------
    This example is only valid to benchmark different implementations, but does not
    represent a use case.

    >>> num_states_in_period = 10000
    >>> num_draws = 500

    >>> delta = np.array(0.9)
    >>> wages = np.random.randn(num_states_in_period, 2)
    >>> rewards = np.random.randn(num_states_in_period, 4)
    >>> draws = np.random.randn(num_draws, 4)
    >>> emaxs = np.random.randn(num_states_in_period, 4)

    >>> get_continuation_value(wages, rewards, draws, emaxs, delta).shape
    (10000, 4, 500)

    """
    num_draws = draws.shape[0]
    num_choices = nonpec.shape[0]
    num_wages = wages.shape[0]

    for i in range(num_draws):
        for j in range(num_choices):
            if j < num_wages:
                rew_ex = wages[j] * draws[i, j] + nonpec[j]
            else:
                rew_ex = nonpec[j] + draws[i, j]

            cont_value = rew_ex + delta * emaxs[j]

            if j == 2 and max_education:
                cont_value += INADMISSIBILITY_PENALTY

            rew_ex_post[j, i] = rew_ex
            continuation_value[j, i] = cont_value
