""" This module contains all the capabilities to solve the dynamic
programming problem.
"""

# standard library
import numpy as np

# project library
from robupy.checks._checks_solve import _checks
from robupy.ambiguity import *
from robupy.risk import *

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

    ambiguity = robupy_obj.get_attr('ambiguity')

    level = ambiguity['level']

    measure = ambiguity['measure']

    para = ambiguity['para']

    # Create grid of possible/admissible state space values
    states_all, states_number_period, mapping_state_idx = _create_state_space(
        robupy_obj)

    # Run checks of the state space variables
    if debug is True:
        _checks('state_space', robupy_obj, states_all, states_number_period,
                mapping_state_idx)

    # Draw random variables
    np.random.seed(seed)

    eps_standard_periods = np.random.multivariate_normal(np.zeros(4), np.identity(4),
                                                 (num_periods, num_draws))

    # Handle special case. This case is useful to perform hand
    # calculations for testing.
    all_zeros = (np.count_nonzero(init_dict['SHOCKS']) == 0)

    if all_zeros:
        true_cholesky = np.zeros((4, 4))
    else:
        true_cholesky = np.linalg.cholesky(init_dict['SHOCKS'])

    # Initialize container for expected future values
    emax = np.tile(np.nan, (num_periods, max(states_number_period)))

    # Initialize components of payoffs. The ex post payoffs and the future payoffs
    # are only stored for debugging reasons and can be removed if memory usage
    # turns out to be an issue.
    period_payoffs_ex_ante = np.tile(np.nan, (
        num_periods, max(states_number_period), 4))
    period_payoffs_ex_post = np.tile(np.nan, (
        num_periods, max(states_number_period), 4))
    future_payoffs = np.tile(np.nan,
                             (num_periods, max(states_number_period), 4))

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

            covars = [1.0, edu + edu_start, exp_A, exp_A ** 2, exp_B,
                      exp_B ** 2]

            # Calculate systematic part of wages in occupation A
            period_payoffs_ex_ante[period, k, 0] = np.exp(
                np.dot(coeffs['A'], covars))

            # Calculate systematic part pf wages in occupation B
            period_payoffs_ex_ante[period, k, 1] = np.exp(
                np.dot(coeffs['B'], covars))

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

    # Initialize baseline disturbances
    eps_baseline_periods = np.tile(np.nan, (num_periods, num_draws, 4))

    for period in range(num_periods):
        eps_baseline_periods[period, :, :] = np.dot(true_cholesky,
                                            eps_standard_periods[period, :,
                                            :].T).T

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Extract disturbances
        eps_baseline = eps_baseline_periods[period, :, :]
        eps_standard = eps_standard_periods[period, :, :]

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Simulate the expected future value
            if level == 0.00:

                simulated, period_payoffs_ex_ante, \
                period_payoffs_ex_post, future_payoffs = \
                    simulate_emax_risk(num_draws, period_payoffs_ex_post,
                        eps_baseline_periods, period,
                                                k, period_payoffs_ex_ante, edu_max,
                                                edu_start, mapping_state_idx,
                                                states_all, future_payoffs,
                                                num_periods, emax)
            else:
                print('test')
                # TODO: How to calculate the period_payoff_ex_post, etc
                simulated = simulate_emax_ambiguity(num_draws, period_payoffs_ex_post,
                                      eps_standard, period,
                   k, period_payoffs_ex_ante, edu_max, edu_start,
                   mapping_state_idx, states_all, future_payoffs,
                   num_periods, emax, ambiguity, true_cholesky)

            # Collect
            emax[period, k] = simulated

    # Run checks on expected future values and its ingredients
    if debug is True:
        _checks('emax', robupy_obj, states_all, states_number_period, emax,
                future_payoffs)

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
                                         min(num_periods,
                                             (edu_max - edu_start + 1)), 2))

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
                        states_all[period, k, :] = [exp_A, exp_B, edu,
                                                    edu_lagged]

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