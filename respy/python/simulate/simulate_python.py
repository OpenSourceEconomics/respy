import numpy as np

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
import pandas as pd
from respy.python.shared.shared_auxiliary import get_total_values


def pyth_simulate(
    num_periods,
    num_agents_sim,
    states,
    states_indexer,
    periods_draws_sims,
    seed_sim,
    file_sim,
    edu_spec,
    optim_paras,
    num_types,
    is_debug,
):
    """ Wrapper for PYTHON and F2PY implementation of sample simulation.

    Parameters
    ----------
    num_periods : int
    num_agents_sim : int
    states : pd.DataFrame
    states_indexer : np.array
    periods_draws_sims : ???
    seed_sim : ???
    file_sim : ???
    edu_spec : dict
    optim_paras : dict
    num_types : int
    is_debug : bool

    Returns
    -------
    simulated_data : pd.DataFrame

    """
    record_simulation_start(num_agents_sim, seed_sim, file_sim)

    # Standard deviates transformed to the distributions relevant for the agents actual
    # decision making as traversing the tree.
    periods_draws_sims_transformed = np.tile(
        np.nan, (num_periods, num_agents_sim, 4)
    )

    for period in range(num_periods):
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
        num_types, optim_paras, num_agents_sim, initial_education, is_debug
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
            ]
        )

        record_simulation_progress(i, file_sim)

        for period in range(num_periods):

            exp_a, exp_b, edu, choice_lagged, type_ = current_state

            state_idx = states_indexer[
                period, exp_a, exp_b, edu, choice_lagged - 1, type_
            ]
            state = states.iloc[state_idx]

            # Select relevant subset
            draws = periods_draws_sims_transformed[period, i, :]

            # Get total value of admissible states
            total_values, rewards_ex_post = get_total_values(
                state, draws, optim_paras
            )

            # We need to ensure that no individual chooses an inadmissible state. This
            # cannot be done directly in the get_total_values function as the penalty
            # otherwise dominates the interpolation equation. The parameter
            # INADMISSIBILITY_PENALTY is a compromise. It is only relevant in very
            # constructed cases.
            if edu >= edu_spec["max"]:
                total_values[2] = -HUGE_FLOAT

            # Determine optimal choice
            max_idx = np.argmax(total_values)

            # Record wages
            wages = np.array([state.wage_a, state.wage_b])
            wage = (
                wages[max_idx] * draws[max_idx]
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
                "Systematic_Reward_1": state.rewards_systematic_a,
                "Systematic_Reward_2": state.rewards_systematic_b,
                "Systematic_Reward_3": state.rewards_systematic_edu,
                "Systematic_Reward_4": state.rewards_systematic_home,
                "Shock_Reward_1": draws[0],
                "Shock_Reward_2": draws[1],
                "Shock_Reward_3": draws[2],
                "Shock_Reward_4": draws[3],
                "Discount_Rate": optim_paras["delta"],
                # For testing purposes, we also explicitly include the general reward
                # component, the common component, and the immediate ex post rewards.
                "General_Reward_1": state.rewards_general_a,
                "General_Reward_2": state.rewards_general_b,
                "Common_Reward": state.rewards_common,
                "Immediate_Reward_1": rewards_ex_post[0],
                "Immediate_Reward_2": rewards_ex_post[1],
                "Immediate_Reward_3": rewards_ex_post[2],
                "Immediate_Reward_4": rewards_ex_post[3],
            }
            data.append(row)

    record_simulation_stop(file_sim)

    simulated_data = pd.DataFrame.from_records(data)

    return simulated_data
