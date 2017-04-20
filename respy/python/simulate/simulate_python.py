import numpy as np

from respy.python.record.record_simulation import record_simulation_progress
from respy.python.record.record_simulation import record_simulation_stop
from respy.python.record.record_simulation import record_simulation_start
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.simulate.simulate_auxiliary import get_random_types
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_constants import MISSING_FLOAT


def pyth_simulate(periods_rewards_systematic, mapping_state_idx, periods_emax,
        states_all, num_periods, edu_start, edu_max, num_agents_sim,
        periods_draws_sims, seed_sim, file_sim, optim_paras,
        num_types, type_spec, is_debug):
    """ Wrapper for PYTHON and F2PY implementation of sample simulation.
    """

    record_simulation_start(num_agents_sim, seed_sim, file_sim)

    # Standard deviates transformed to the distributions relevant for
    # the agents actual decision making as traversing the tree.
    periods_draws_sims_transformed = np.tile(np.nan,
        (num_periods, num_agents_sim, 4))

    for period in range(num_periods):
        periods_draws_sims_transformed[period, :, :] = transform_disturbances(
            periods_draws_sims[period, :, :], np.array([0.0, 0.0, 0.0, 0.0]),
            optim_paras['shocks_cholesky'])

    # We also need to sample the set of initial conditions.
    types = get_random_types(num_types, type_spec, num_agents_sim, is_debug)

    # Simulate agent experiences
    count = 0

    # Initialize data
    dataset = np.tile(MISSING_FLOAT, (num_agents_sim * num_periods, 22))

    for i in range(num_agents_sim):

        current_state = states_all[0, 0, :].copy()

        # We need to modify the initial conditions
        current_state[-1] = types[i]

        record_simulation_progress(i, file_sim)

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_a, exp_b, edu, edu_lagged, type_ = current_state
            k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged, type_]

            # Write agent identifier and current period to data frame
            dataset[count, :2] = i, period

            # Select relevant subset
            rewards_systematic = periods_rewards_systematic[period, k, :]
            draws = periods_draws_sims_transformed[period, i, :]

            # Get total value of admissible states
            total_values = get_total_values(period, num_periods, optim_paras,
                rewards_systematic, draws, edu_max, edu_start,
                mapping_state_idx, periods_emax, k, states_all)

            # Determine optimal choice
            max_idx = np.argmax(total_values)

            # Record agent decision
            dataset[count, 2] = max_idx + 1

            # Record wages
            dataset[count, 3] = MISSING_FLOAT
            if max_idx in [0, 1]:
                dataset[count, 3] = rewards_systematic[max_idx] * draws[max_idx]

            # Write relevant state space for period to data frame. However,
            # the individual's type is not part of the observed dataset. This
            # is included in the simulated dataset.
            dataset[count, 4:8] = current_state[:4]

            # Special treatment for education
            dataset[count, 6] += edu_start

            # As we are working with a simulated dataset, we can also output
            # additional information that is not available in an observed
            # dataset. The discount rate is included as this allows to
            # construct the EMAX with the information provided in the
            # simulation output.
            dataset[count,  8:9] = type_
            dataset[count,  9:13] = total_values
            dataset[count, 13:17] = rewards_systematic
            dataset[count, 17:21] = draws
            dataset[count, 21:22] = optim_paras['delta']

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
