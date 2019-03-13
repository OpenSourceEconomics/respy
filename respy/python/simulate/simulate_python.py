import numpy as np
import pandas as pd

from respy.python.record.record_simulation import record_simulation_progress
from respy.python.simulate.simulate_auxiliary import (
    get_random_choice_lagged_start,
)
from respy.python.record.record_simulation import record_simulation_start
from respy.python.simulate.simulate_auxiliary import get_random_edu_start
from respy.python.record.record_simulation import record_simulation_stop
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.simulate.simulate_auxiliary import get_random_types
from respy.python.shared.shared_constants import (
    HUGE_FLOAT,
    DATA_LABELS_SIM,
    DATA_FORMATS_SIM,
)
from respy.python.shared.shared_auxiliary import get_continuation_value


def pyth_simulate(
    state_space,
    num_agents_sim,
    periods_draws_sims,
    seed_sim,
    file_sim,
    edu_spec,
    optim_paras,
    is_debug,
):
    """ Wrapper for PYTHON and F2PY implementation of sample simulation.

    Parameters
    ----------
    state_space : class
        Class of state space.
    num_agents_sim : int
        Number of simulated agents.
    periods_draws_sims : np.ndarray
        Array with shape (num_periods, num_agents_sim, num_choices)
    seed_sim : int
        Seed for the simulation.
    file_sim : ???
    edu_spec : dict
    optim_paras : dict
    is_debug : bool
        Flag for debugging modus.

    Returns
    -------
    simulated_data : pd.DataFrame
        Dataset of simulated agents.

    """
    record_simulation_start(num_agents_sim, seed_sim, file_sim)

    # Standard deviates transformed to the distributions relevant for the agents actual
    # decision making as traversing the tree.
    periods_draws_sims_transformed = np.full(
        (state_space.num_periods, num_agents_sim, 4), np.nan
    )

    for period in range(state_space.num_periods):
        periods_draws_sims_transformed[period, :, :] = transform_disturbances(
            periods_draws_sims[period, :, :],
            np.zeros(4),
            optim_paras["shocks_cholesky"],
        )

    # We also need to sample the set of initial conditions.
    initial_education = get_random_edu_start(
        edu_spec, num_agents_sim, is_debug
    )
    initial_types = get_random_types(
        state_space.num_types,
        optim_paras,
        num_agents_sim,
        initial_education,
        is_debug,
    )
    initial_choice_lagged = get_random_choice_lagged_start(
        edu_spec, num_agents_sim, initial_education, is_debug
    )

    # We need to modify the initial conditions: (1) Schooling when entering the model
    # and (2) individual type. We need to determine the initial value for the lagged
    # variable.
    current_states = np.c_[
        np.zeros((num_agents_sim, 2)),
        initial_education,
        initial_choice_lagged,
        initial_types,
    ].astype(np.uint8)

    data = []

    for period in range(state_space.num_periods):

        ks = state_space.indexer[
            np.full(num_agents_sim, period),
            current_states[:, 0],
            current_states[:, 1],
            current_states[:, 2],
            current_states[:, 3] - 1,
            current_states[:, 4],
        ]

        # Select relevant subset
        draws = periods_draws_sims_transformed[period]

        # Get total value of admissible states
        total_values, rewards_ex_post = get_continuation_value(
            state_space.rewards[ks, -2:],
            state_space.rewards[ks, :4],
            draws.reshape(-1, 1, 4),
            state_space.emaxs[ks, :4],
            optim_paras["delta"],
        )

        total_values = total_values.reshape(-1, 4)
        rewards_ex_post = rewards_ex_post.reshape(-1, 4)

        # We need to ensure that no individual chooses an inadmissible state. This
        # cannot be done directly in the get_continuation_value function as the penalty
        # otherwise dominates the interpolation equation. The parameter
        # INADMISSIBILITY_PENALTY is a compromise. It is only relevant in very
        # constructed cases.
        total_values[:, 2] = np.where(
            current_states[:, 2] >= edu_spec["max"],
            -HUGE_FLOAT,
            total_values[:, 2],
        )

        # Determine optimal choice
        max_idx = np.argmax(total_values, axis=1)

        # Record wages. Expand matrix with NaNs for choice 2 and 3 for easier indexing.
        wages = (
            np.c_[
                state_space.rewards[ks, -2:],
                np.full((num_agents_sim, 2), np.nan),
            ]
            * draws
        )
        # Do not swap np.arange with : (https://stackoverflow.com/a/46425896/7523785)!
        wage = wages[np.arange(num_agents_sim), max_idx]

        rows = np.c_[
            np.arange(num_agents_sim),
            np.full(num_agents_sim, period),
            max_idx + 1,
            wage,
            # Write relevant state space for period to data frame. However, the
            # individual's type is not part of the observed dataset. This is included in
            # the simulated dataset.
            current_states,
            # As we are working with a simulated dataset, we can also output additional
            # information that is not available in an observed dataset. The discount
            # rate is included as this allows to construct the EMAX with the information
            # provided in the simulation output.
            total_values,
            state_space.rewards[ks, :4],
            draws,
            np.full(num_agents_sim, optim_paras["delta"][0]),
            # For testing purposes, we also explicitly include the general reward
            # component, the common component, and the immediate ex post rewards.
            state_space.rewards[ks, 4:7],
            rewards_ex_post,
        ]

        data.append(rows)

        # Update work experiences or education
        current_states[np.arange(num_agents_sim), max_idx] = np.where(
            max_idx <= 2,
            current_states[np.arange(num_agents_sim), max_idx] + 1,
            current_states[np.arange(num_agents_sim), max_idx],
        )

        # Update lagged activity variable.
        current_states[:, 3] = max_idx + 1

    simulated_data = (
        pd.DataFrame.from_records(np.vstack(data), columns=DATA_LABELS_SIM)
        .astype(DATA_FORMATS_SIM)
        .sort_values(["Identifier", "Period"])
    )

    # TODO: Replace logging which is useless here and kept only for successful testing.
    for i in range(num_agents_sim):
        record_simulation_progress(i, file_sim)
    record_simulation_stop(file_sim)

    return simulated_data
