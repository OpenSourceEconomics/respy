""" This module contains the PYTHON implementations fo several functions
where FORTRAN alternatives are available.
"""
import numpy as np


def simulate_emax(num_periods, num_draws, period, k, eps_relevant,
        payoffs_ex_ante, edu_max, edu_start, emax, states_all,
        mapping_state_idx, delta):
    """ Simulate expected future value.
    """

    # Initialize containers
    payoffs_ex_post, emax_simulated = np.tile(np.nan, 4), 0.0
    future_payoffs = 0.0

    # Calculate maximum value
    for i in range(num_draws):

        # Calculate ex post payoffs
        for j in [0, 1]:
            payoffs_ex_post[j] = payoffs_ex_ante[j] * eps_relevant[i, j]

        for j in [2, 3]:
            payoffs_ex_post[j] = payoffs_ex_ante[j] + eps_relevant[i, j]

        # Check applicability
        if period != (num_periods - 1):

            # Get future values
            future_payoffs = get_future_payoffs(edu_max, edu_start,
                                mapping_state_idx, period, emax, k, states_all)

        # Calculate total utilities
        total_payoffs = payoffs_ex_post + delta * future_payoffs

        # Determine optimal choice
        maximum = max(total_payoffs)

        # Recording expected future value
        emax_simulated += maximum

    # Scaling
    emax_simulated = emax_simulated / num_draws

    # Finishing
    return emax_simulated, payoffs_ex_post, future_payoffs


def get_future_payoffs(edu_max, edu_start, mapping_state_idx, period, emax, k,
        states_all):
    """ Get future payoffs for additional choices.
    """

    # Distribute state space
    exp_A, exp_B, edu, edu_lagged = states_all[period, k, :]

    # Future utilities
    future_payoffs = np.tile(np.nan, 4)

    # Working in occupation A
    future_idx = mapping_state_idx[period + 1, exp_A + 1, exp_B, edu, 0]
    future_payoffs[0] = emax[period + 1, future_idx]

    # Working in occupation B
    future_idx = mapping_state_idx[period + 1, exp_A, exp_B + 1, edu, 0]
    future_payoffs[1] = emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year
    # of schooling is only possible for those that have strictly
    # less than the maximum level of additional education allowed.
    if edu < edu_max - edu_start:
        future_idx = mapping_state_idx[period + 1, exp_A, exp_B, edu + 1, 1]
        future_payoffs[2] = emax[period + 1, future_idx]
    else:
        future_payoffs[2] = -np.inf

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_A, exp_B, edu, 0]
    future_payoffs[3] = emax[period + 1, future_idx]

    # Ensuring that schooling does not increase beyond the
    # maximum allowed level. This is necessary as in the
    # special case where delta is equal to zero,
    # (-np.inf * 0) evaluates to NAN.
    if edu >= edu_max - edu_start:
        future_payoffs[2] = -np.inf

    # Finishing
    return future_payoffs

def calculate_payoffs_ex_ante(num_periods, states_number_period, states_all,
                                edu_start, coeffs_A, coeffs_B, coeffs_edu,
                                coeffs_home, max_states_period):
    """ Calculate ex ante payoffs.
    """

    # Initialize
    period_payoffs_ex_ante = np.tile(np.nan, (num_periods, max_states_period,
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
            period_payoffs_ex_ante[period, k, 0] = np.exp(
                np.dot(coeffs_A, covars))

            # Calculate systematic part pf wages in occupation B
            period_payoffs_ex_ante[period, k, 1] = np.exp(
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

            period_payoffs_ex_ante[period, k, 2] = payoff

            # Calculate systematic part of HOME
            period_payoffs_ex_ante[period, k, 3] = coeffs_home[0]

    # Finishing
    return period_payoffs_ex_ante


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
