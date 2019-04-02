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
from respy.python.shared.shared_constants import HUGE_FLOAT
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

    data = []

    for i in range(num_agents_sim):

        # We need to modify the initial conditions: (1) Schooling when entering the
        # model and (2) individual type. We need to determine the initial value for the
        # lagged variable.
        current_state = np.array(
            [
                0,
                0,
                initial_education[i],
                initial_choice_lagged[i],
                initial_types[i],
            ],
            dtype=np.uint8,
        )

        record_simulation_progress(i, file_sim)

        for period in range(state_space.num_periods):

            exp_a, exp_b, edu, choice_lagged, type_ = current_state

            k = state_space.indexer[
                period, exp_a, exp_b, edu, choice_lagged - 1, type_
            ]

            # Select relevant subset
            draws = periods_draws_sims_transformed[period, i, :]

            # Get total value of admissible states
            total_values, rewards_ex_post = get_continuation_value(
                state_space.rewards[k, -2:],
                state_space.rewards[k, :4],
                draws.reshape(1, -1),
                state_space.emaxs[k, :4],
                optim_paras["delta"],
            )

            total_values = total_values.ravel()
            rewards_ex_post = rewards_ex_post.ravel()

            # We need to ensure that no individual chooses an inadmissible state. This
            # cannot be done directly in the get_continuation_value function as the
            # penalty otherwise dominates the interpolation equation. The parameter
            # INADMISSIBILITY_PENALTY is a compromise. It is only relevant in very
            # constructed cases.
            if edu >= edu_spec["max"]:
                total_values[2] = -HUGE_FLOAT

            # Determine optimal choice
            max_idx = np.argmax(total_values)

            # Record wages
            wage = (
                state_space.rewards[k, -2:][max_idx] * draws[max_idx]
                if max_idx in [0, 1]
                else np.nan
            )

            # Update work experiences or education
            if max_idx in [0, 1, 2]:
                current_state[max_idx] += 1

            # Update lagged activity variable.
            current_state[3] = max_idx + 1

            row = {
                "Identifier": i,
                "Period": period,
                "Choice": max_idx + 1,
                "Wage": wage,
                # Write relevant state space for period to data frame. However, the
                # individual's type is not part of the observed dataset. This is
                # included in the simulated dataset.
                "Experience_A": exp_a,
                "Experience_B": exp_b,
                "Years_Schooling": edu,
                "Lagged_Choice": choice_lagged,
                # As we are working with a simulated dataset, we can also output
                # additional information that is not available in an observed dataset.
                # The discount rate is included as this allows to construct the EMAX
                # with the information provided in the simulation output.
                "Type": type_,
                "Total_Reward_1": total_values[0],
                "Total_Reward_2": total_values[1],
                "Total_Reward_3": total_values[2],
                "Total_Reward_4": total_values[3],
                "Systematic_Reward_1": state_space.rewards[k, 0],
                "Systematic_Reward_2": state_space.rewards[k, 1],
                "Systematic_Reward_3": state_space.rewards[k, 2],
                "Systematic_Reward_4": state_space.rewards[k, 3],
                "Shock_Reward_1": draws[0],
                "Shock_Reward_2": draws[1],
                "Shock_Reward_3": draws[2],
                "Shock_Reward_4": draws[3],
                "Discount_Rate": optim_paras["delta"][0],
                # For testing purposes, we also explicitly include the general reward
                # component, the common component, and the immediate ex post rewards.
                "General_Reward_1": state_space.rewards[k, 4],
                "General_Reward_2": state_space.rewards[k, 5],
                "Common_Reward": state_space.rewards[k, 6],
                "Immediate_Reward_1": rewards_ex_post[0],
                "Immediate_Reward_2": rewards_ex_post[1],
                "Immediate_Reward_3": rewards_ex_post[2],
                "Immediate_Reward_4": rewards_ex_post[3],
            }
            data.append(row)

    record_simulation_stop(file_sim)

    simulated_data = pd.DataFrame.from_records(data, columns=data[0].keys())

    return simulated_data
