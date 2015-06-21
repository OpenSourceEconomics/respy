""" This module contains all the capabilities to solve the dynamic
programming problem.
"""

# standard library
import numpy as np

# project library
from robupy._checks_solve import _checks
from robupy._shared import _get_future_payoffs

''' Public function
'''


def solve(robupy_obj):
    """ Solve dynamic programming problem by backward induction.
    """
    # Antibugging
    assert (robupy_obj.get_status())

    # Distribute class attributes
    init_dict = robupy_obj.get_attr('init_dict')

    delta = robupy_obj.get_attr('delta')

    num_periods = robupy_obj.get_attr('num_periods')

    num_draws = robupy_obj.get_attr('num_draws')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    seed = robupy_obj.get_attr('seed')

    debug = robupy_obj.get_attr('debug')

    # Create grid of possible/admissible state space values
    states_all, states_number_period, mapping_state_idx = _create_state_space(robupy_obj)

    # Run checks of the state space variables
    if debug is True:
        _checks('state_space', robupy_obj, states_all, states_number_period, mapping_state_idx)

    # Draw random variables
    np.random.seed(seed)

    eps = np.random.multivariate_normal(np.zeros(4), init_dict['SHOCKS'],
                                        (num_periods, num_draws))

    # Initialize container for expected future values
    emax = np.tile(np.nan, (num_periods, max(states_number_period)))

    # Initialize components of payoffs. The ex post payoffs and the future payoffs
    # are only stored for debugging reasons and can be removed if memory usage
    # turns out to be an issue.
    period_payoffs_ex_ante = np.tile(np.nan, (num_periods, max(states_number_period), 4))
    period_payoffs_ex_post = np.tile(np.nan, (num_periods, max(states_number_period), 4))
    future_payoffs = np.tile(np.nan, (num_periods, max(states_number_period), 4))

    # Calculate systematic instantaneous payoffs
    for period in range(num_periods - 1, -1, -1):

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Distribute state space
            exp_A, exp_B, edu, edu_lagged = states_all[period, k, :]

            # Auxiliary objects
            coeffs = dict()
            coeffs['A'] = [init_dict['A']['int']] + init_dict['A']['coeff']
            coeffs['B'] = [init_dict['B']['int']] + init_dict['B']['coeff']

            covars = [1.0, edu, exp_A, exp_A ** 2, exp_B, exp_B ** 2]

            # Calculate systematic part of wages in occupation A
            period_payoffs_ex_ante[period, k, 0] = np.exp(np.dot(coeffs['A'], covars))

            # Calculate systematic part pf wages in occupation B
            period_payoffs_ex_ante[period, k, 1] = np.exp(np.dot(coeffs['B'], covars))

            # Calculate systematic part of schooling utility
            payoff = init_dict['EDUCATION']['int']

            # Tuition cost for higher education if agents move
            # beyond high school.
            if edu + edu_start > 12:
                payoff -= init_dict['EDUCATION']['coeff'][0]
            # Psychic cost of going back to school
            if edu_lagged == 0:
                payoff -= init_dict['EDUCATION']['coeff'][1]

            period_payoffs_ex_ante[period, k, 2] = payoff

            # Calculate systematic part of HOME
            period_payoffs_ex_ante[period, k, 3] = init_dict['HOME']['int']

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Initialize container
            emax[period, k] = 0

            # Calculate maximum value
            for i in range(num_draws):

                # Calculate ex post payoffs
                for j in [0, 1]:
                    period_payoffs_ex_post[period, k, j] = period_payoffs_ex_ante[period, k, j] * \
                        np.exp(eps[period, i, j])

                for j in [2, 3]:
                    period_payoffs_ex_post[period, k, j] = period_payoffs_ex_ante[period, k, j] + \
                        eps[period, i, j]

                # Calculate future utilities
                future_payoffs[period, k, :] = _get_future_payoffs(edu_max, edu_start, mapping_state_idx, period, emax,
                                                     k, states_all, num_periods, delta)

                # Calculate total utilities
                total_payoffs = period_payoffs_ex_post[period, k, :] + future_payoffs[period, k, :]

                # Determine optimal choice
                maximum = max(total_payoffs)

                # Recording expected future value
                emax[period, k] += maximum

            # Scaling
            emax[period, k] = emax[period, k] / num_draws

    # Run checks on expected future values and its ingredients
    if debug is True:
        _checks('emax', robupy_obj, states_all, states_number_period, emax, future_payoffs)

    # Update
    robupy_obj.unlock()

    robupy_obj.set_attr('states_number_period', states_number_period)

    robupy_obj.set_attr('period_payoffs_ex_ante', period_payoffs_ex_ante)

    robupy_obj.set_attr('period_payoffs_ex_post', period_payoffs_ex_post)

    robupy_obj.set_attr('mapping_state_idx', mapping_state_idx)

    robupy_obj.set_attr('future_payoffs', future_payoffs)

    robupy_obj.set_attr('states_all', states_all)

    robupy_obj.set_attr('emax', emax)

    robupy_obj.lock()

    # Finishing
    return robupy_obj


''' Private functions
'''


def _create_state_space(robupy_obj):
    """ Create grid for state space.
    """

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    # Array for possible realization of state space by period
    states_all = np.tile(np.nan, (num_periods, 100000, 4))

    # Array for maximum number of realizations of state space by period
    states_number_period = np.tile(np.nan, num_periods)

    # Array for the mapping of state space values to indices in variety
    # of matrices.
    mapping_state_idx = np.tile(np.nan, (num_periods, num_periods, num_periods,
                            min(num_periods, (edu_max - edu_start + 1)), 2))

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
                        if total > period: continue

                        # Collect all possible realizations of state space
                        states_all[period, k, :] = [exp_A, exp_B, edu, edu_lagged]

                        # Collect mapping of state space to array index.
                        mapping_state_idx[period, exp_A, exp_B, edu,
                                edu_lagged] = k

                        # Update count
                        k += 1

        # Record maximum number of state space realizations by time period
        states_number_period[period] = k

    # Type transformation
    states_number_period = np.array(states_number_period, dtype='int')

    # Cutting to size
    states_all = states_all[:, :max(states_number_period), :]

    # Finishing
    return states_all, states_number_period, mapping_state_idx
