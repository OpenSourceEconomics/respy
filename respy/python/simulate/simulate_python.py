import numpy as np

from respy.python.record.record_simulation import record_simulation_progress
from respy.python.record.record_simulation import record_simulation_stop
from respy.python.record.record_simulation import record_simulation_start
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.shared.shared_auxiliary import get_total_values


def pyth_simulate(periods_rewards_systematic, mapping_state_idx, periods_emax,
        states_all, num_periods, edu_start, edu_max, delta, num_agents_sim,
        periods_draws_sims, seed_sim, file_sim, model_paras):
    """ Wrapper for PYTHON and F2PY implementation of sample simulation.
    """

    record_simulation_start(num_agents_sim, seed_sim, file_sim)

    # Standard deviates transformed to the distributions relevant for
    # the agents actual decision making as traversing the tree.
    periods_draws_sims_transformed = np.tile(np.nan,
        (num_periods, num_agents_sim, 4))

    for period in range(num_periods):
        periods_draws_sims_transformed[period, :, :] = transform_disturbances(
            periods_draws_sims[period, :, :], model_paras['shocks_cholesky'])

    # Simulate agent experiences
    count = 0

    # Initialize data
    dataset = np.tile(MISSING_FLOAT, (num_agents_sim * num_periods, 8))

    for i in range(num_agents_sim):

        current_state = states_all[0, 0, :].copy()

        dataset[count, 0] = i

        record_simulation_progress(i, file_sim)

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_a, exp_b, edu, edu_lagged = current_state

            k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]

            # Write agent identifier and current period to data frame
            dataset[count, :2] = i, period

            # Select relevant subset
            rewards_systematic = periods_rewards_systematic[period, k, :]
            draws = periods_draws_sims_transformed[period, i, :]

            # Get total value of admissible states
            total_values = get_total_values(period,
                num_periods, delta, rewards_systematic, draws, edu_max,
                edu_start, mapping_state_idx, periods_emax, k, states_all)

            # Determine optimal choice
            max_idx = np.argmax(total_values)

            # Record agent decision
            dataset[count, 2] = max_idx + 1

            # Record wages
            dataset[count, 3] = MISSING_FLOAT
            if max_idx in [0, 1]:
                dataset[count, 3] = rewards_systematic[max_idx] * draws[max_idx]

            # Write relevant state space for period to data frame
            dataset[count, 4:8] = current_state

            # Special treatment for education
            dataset[count, 6] += edu_start

            # Update work experiences and education
            if max_idx == 0:
                current_state[0] += 1
            elif max_idx == 1:
                current_state[1] += 1
            elif max_idx == 2:
                current_state[2] += 1

            # Update lagged education
            current_state[3] = 0

            if max_idx == 2:
                current_state[3] = 1

            # Update row indicator
            count += 1

    record_simulation_stop(file_sim)

    # Finishing
    return dataset
