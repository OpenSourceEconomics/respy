import numpy as np

from respy.python.record.record_simulation import record_simulation_progress
from respy.python.shared.shared_auxiliary import back_out_systematic_wages
from respy.python.shared.shared_auxiliary import calculate_rewards_general
from respy.python.shared.shared_auxiliary import calculate_rewards_common
from respy.python.record.record_simulation import record_simulation_start
from respy.python.simulate.simulate_auxiliary import get_random_edu_start
from respy.python.record.record_simulation import record_simulation_stop
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.simulate.simulate_auxiliary import get_random_types
from respy.python.shared.shared_auxiliary import construct_covariates
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT


def pyth_simulate(periods_rewards_systematic, mapping_state_idx, periods_emax, states_all,
        num_periods, num_agents_sim, periods_draws_sims, seed_sim, file_sim, edu_spec, optim_paras,
        num_types, is_debug):
    """ Wrapper for PYTHON and F2PY implementation of sample simulation.
    """

    record_simulation_start(num_agents_sim, seed_sim, file_sim)

    # Standard deviates transformed to the distributions relevant for the agents actual decision
    # making as traversing the tree.
    periods_draws_sims_transformed = np.tile(np.nan, (num_periods, num_agents_sim, 4))

    for period in range(num_periods):
        periods_draws_sims_transformed[period, :, :] = transform_disturbances(
            periods_draws_sims[period, :, :], np.array([0.0, 0.0, 0.0, 0.0]),
            optim_paras['shocks_cholesky'])

    # We also need to sample the set of initial conditions.
    edu_start = get_random_edu_start(edu_spec, num_agents_sim, is_debug)
    types = get_random_types(num_types, optim_paras, num_agents_sim, edu_start, is_debug)

    # Simulate agent experiences
    count = 0

    # Initialize data
    dataset = np.tile(MISSING_FLOAT, (num_agents_sim * num_periods, 26))

    for i in range(num_agents_sim):

        current_state = states_all[0, 0, :].copy()

        # We need to modify the initial conditions: (1) Schooling when entering the model and (2)
        # individual type.
        current_state[2] = edu_start[i]
        current_state[4] = types[i]

        record_simulation_progress(i, file_sim)

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_a, exp_b, edu, activity_lagged, type_ = current_state
            k = mapping_state_idx[period, exp_a, exp_b, edu, activity_lagged, type_]

            # Write agent identifier and current period to data frame
            dataset[count, :2] = i, period

            # Select relevant subset
            rewards_systematic = periods_rewards_systematic[period, k, :]
            draws = periods_draws_sims_transformed[period, i, :]

            # Get total value of admissible states
            total_values = get_total_values(period, num_periods, optim_paras, rewards_systematic,
                draws, edu_spec, mapping_state_idx, periods_emax, k, states_all)

            # We need to ensure that no individual chooses an inadmissible state. This cannot be
            # done directly in the get_total_values function as the penalty otherwise dominates
            # the interpolation equation. The parameter INADMISSIBILITY_PENALTY is a compromise.
            # It is only relevant in very constructed cases.
            if edu >= edu_spec['max']:
                total_values[2] = -HUGE_FLOAT

            # Determine optimal choice
            max_idx = np.argmax(total_values)

            # Record agent decision
            dataset[count, 2] = max_idx + 1

            # Record wages
            dataset[count, 3] = MISSING_FLOAT
            wages_systematic = back_out_systematic_wages(rewards_systematic, exp_a, exp_b,
                edu, activity_lagged, optim_paras)

            if max_idx in [0, 1]:
                dataset[count, 3] = wages_systematic[max_idx] * draws[max_idx]

            # Write relevant state space for period to data frame. However, the individual's type
            # is not part of the observed dataset. This is included in the simulated dataset.
            dataset[count, 4:8] = current_state[:4]

            # As we are working with a simulated dataset, we can also output additional
            # information that is not available in an observed dataset. The discount rate is
            # included as this allows to construct the EMAX with the information provided in the
            # simulation output.
            dataset[count,  8:9] = type_
            dataset[count,  9:13] = total_values
            dataset[count, 13:17] = rewards_systematic
            dataset[count, 17:21] = draws
            dataset[count, 21:22] = optim_paras['delta']

            # For testing purposes, we also explicitly include the general reward component and
            # the common component. We also include an internal version of the wage variables as
            # otherwise it gets truncated to two digits which does make some unit tests untenable.
            covariates = construct_covariates(exp_a, exp_b, edu, activity_lagged, type_, period)
            dataset[count, 22:24] = calculate_rewards_general(covariates, optim_paras)
            dataset[count, 24:25] = calculate_rewards_common(covariates, optim_paras)
            dataset[count, 25:26] = dataset[count, 3]

            # Update work experiences and education
            if max_idx == 0:
                current_state[0] += 1
            elif max_idx == 1:
                current_state[1] += 1
            elif max_idx == 2:
                current_state[2] += 1

            # Update lagged activity variable.
            if max_idx == 0:
                current_state[3] = 2
            elif max_idx == 1:
                current_state[3] = 3
            elif max_idx == 2:
                current_state[3] = 1
            else:
                current_state[3] = 0

            # Update row indicator
            count += 1

    record_simulation_stop(file_sim)

    # Finishing
    return dataset
