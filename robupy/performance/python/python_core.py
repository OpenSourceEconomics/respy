""" This module contains the PYTHON implementations fo several functions
where FORTRAN alternatives are available.
"""

# standard library
import numpy as np
import logging

# project libray
from robupy.performance.python.ambiguity import get_payoffs_ambiguity
from robupy.performance.python.risk import get_payoffs_risk
from robupy.performance.python.auxiliary import get_future_payoffs

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')


def backward_induction(num_periods, max_states_period, eps_relevant_periods, num_draws,
            states_number_period, periods_payoffs_ex_ante, edu_max, edu_start,
            mapping_state_idx, states_all, delta, debug, cholesky, level, measure):
    """ Backward iteration procedure.
    """


    with_ambiguity = (level > 0.00)

    # Initialize
    periods_emax = np.tile(-99.00, (num_periods, max_states_period))
    periods_payoffs_ex_post = np.tile(-99.00, (num_periods,
                                               max_states_period, 4))
    periods_future_payoffs = np.tile(-99.00, (num_periods,
                                               max_states_period, 4))

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Logging.
        logger.info('... solving period ' + str(period))

        # Extract disturbances
        eps_relevant = eps_relevant_periods[period, :, :]

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Extract payoffs
            payoffs_ex_ante = periods_payoffs_ex_ante[period, k, :]

            # Simulate the expected future value.
            if with_ambiguity:
                emax, payoffs_ex_post, future_payoffs = \
                    get_payoffs_ambiguity(num_draws, eps_relevant, period, k,
                        payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
                        states_all, num_periods, periods_emax, delta, debug,
                        cholesky, level, measure)
            else:
                emax, payoffs_ex_post, future_payoffs = \
                    get_payoffs_risk(num_draws, eps_relevant, period, k,
                        payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
                        states_all, num_periods, periods_emax, delta)

            # Collect information
            periods_payoffs_ex_post[period, k, :] = payoffs_ex_post
            periods_future_payoffs[period, k, :] = future_payoffs

            # Collect
            periods_emax[period, k] = emax

    # Finishing
    return periods_emax, periods_payoffs_ex_post, periods_future_payoffs


def create_state_space(num_periods, edu_start, edu_max, min_idx):
    """ Create grid for state space.
    """
    # Array for possible realization of state space by period
    states_all = np.tile(-99, (num_periods, 100000, 4))

    # Array for the mapping of state space values to indices in variety
    # of matrices.
    mapping_state_idx = np.tile(-99, (num_periods, num_periods, num_periods,
                                         min_idx, 2))

    # Array for maximum number of realizations of state space by period
    states_number_period = np.tile(np.nan, num_periods)

    # Construct state space by periods
    for period in range(num_periods):

        # Count admissible realizations of state space by period
        k = 0

        # Loop over all admissible work experiences for occupation A
        for exp_A in range(num_periods + 1):

            # Loop over all admissible work experience for occupation B
            for exp_B in range(num_periods + 1):

                # Loop over all admissible additional education levels
                for edu in range(num_periods + 1):

                    # Agent cannot attain more additional education
                    # than (EDU_MAX - EDU_START).
                    if edu > (edu_max - edu_start):
                        continue

                    # Loop over all admissible values for leisure. Note that
                    # the leisure variable takes only zero/value. The time path
                    # does not matter.
                    for edu_lagged in [0, 1]:

                        # Check if lagged education admissible. (1) In the
                        # first period all agents have lagged schooling equal
                        # to one.
                        if (edu_lagged == 0) and (period == 0):
                            continue
                        # (2) Whenever an agent has not acquired any additional
                        # education and we are not in the first period,
                        # then this cannot be the case.
                        if (edu_lagged == 1) and (edu == 0) and (period > 0):
                            continue
                        # (3) Whenever an agent has only acquired additional
                        # education, then edu_lagged cannot be zero.
                        if (edu_lagged == 0) and (edu == period):
                            continue

                        # Check if admissible for time constraints
                        total = edu + exp_A + exp_B

                        # Note that the total number of activities does not
                        # have is less or equal to the total possible number of
                        # activities as the rest is implicitly filled with
                        # leisure.
                        if total > period:
                            continue

                        # Collect all possible realizations of state space
                        states_all[period, k, :] = [exp_A, exp_B, edu,
                                                    edu_lagged]

                        # Collect mapping of state space to array index.
                        mapping_state_idx[period, exp_A, exp_B, edu,
                                          edu_lagged] = k

                        # Update count
                        k += 1

        # Record maximum number of state space realizations by time period
        states_number_period[period] = k

    # Finishing
    return states_all, states_number_period, mapping_state_idx


def calculate_payoffs_ex_ante(num_periods, states_number_period, states_all,
                                edu_start, coeffs_A, coeffs_B, coeffs_edu,
                                coeffs_home, max_states_period):
    """ Calculate ex ante payoffs.
    """

    # Initialize
    periods_payoffs_ex_ante = np.tile(np.nan, (num_periods, max_states_period,
                                                  4))

    # Calculate systematic instantaneous payoffs
    for period in range(num_periods - 1, -1, -1):

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Distribute state space
            exp_A, exp_B, edu, edu_lagged = states_all[period, k, :]

            # Auxiliary objects
            covars = [1.0, edu + edu_start, exp_A, exp_A ** 2, exp_B,
                          exp_B ** 2]

            # Calculate systematic part of wages in occupation A
            periods_payoffs_ex_ante[period, k, 0] = np.exp(
                np.dot(coeffs_A, covars))

            # Calculate systematic part pf wages in occupation B
            periods_payoffs_ex_ante[period, k, 1] = np.exp(
                np.dot(coeffs_B, covars))

            # Calculate systematic part of schooling utility
            payoff = coeffs_edu[0]

            # Tuition cost for higher education if agents move
            # beyond high school.
            if edu + edu_start >= 12:
                payoff += coeffs_edu[1]
            # Psychic cost of going back to school
            if edu_lagged == 0:
                payoff += coeffs_edu[2]

            periods_payoffs_ex_ante[period, k, 2] = payoff

            # Calculate systematic part of HOME
            periods_payoffs_ex_ante[period, k, 3] = coeffs_home[0]

    # Finishing
    return periods_payoffs_ex_ante

def simulate_sample(num_agents, states_all, num_periods,
        mapping_state_idx, periods_payoffs_ex_ante, periods_eps_relevant,
        edu_max, edu_start, periods_emax, delta):
    """ Sample simulation
    """
    count = 0

    # Initialize data
    dataset = np.tile(-99.00, (num_agents * num_periods, 8))

    for i in range(num_agents):

        current_state = states_all[0, 0, :].copy()

        dataset[count, 0] = i

        # Logging
        if (i != 0) and (i % 100 == 0):
            logger.info('... simulated ' + str(i) + ' agents')

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_A, exp_B, edu, edu_lagged = current_state

            k = mapping_state_idx[period, exp_A, exp_B, edu, edu_lagged]

            # Write agent identifier and current period to data frame
            dataset[count, :2] = i, period

            payoffs_ex_post = np.tile(np.nan, 4)

            # Calculate ex post payoffs
            for j in [0, 1]:
                payoffs_ex_post[j] = periods_payoffs_ex_ante[period,
                                                                    k, j] * \
                                                np.exp(periods_eps_relevant[period, i, j])

            for j in [2, 3]:
                payoffs_ex_post[j] = periods_payoffs_ex_ante[period,
                                                                    k, j] + \
                                                periods_eps_relevant[period, i, j]

            # Calculate future utilities
            if period == (num_periods - 1):
                future_payoffs = np.zeros(4)
            else:
                future_payoffs = get_future_payoffs(edu_max, edu_start,
                                                             mapping_state_idx,
                                                             period, periods_emax, k,
                                                             states_all)

            # Calculate total utilities
            total_payoffs = payoffs_ex_post + delta * future_payoffs

            # Determine optimal choice
            max_idx = np.argmax(total_payoffs)

            # Record agent decision
            dataset[count, 2] = max_idx + 1

            # Record earnings
            dataset[count, 3] = -99.00
            if max_idx in [0, 1]:
                dataset[count, 3] = payoffs_ex_post[max_idx]

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

    # Finishing
    return dataset