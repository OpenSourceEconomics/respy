""" This module contains all the capabilities to solve the dynamic
programming problem.
"""

# standard library
import numpy as np
import logging
import shlex
import os

# project library
import robupy.fort.performance as perf

from robupy.checks.checks_solve import checks
from robupy.ambiguity import *
from robupy.risk import *

# Logging
logger = logging.getLogger('ROBUPY')

''' Public function
'''


def solve(robupy_obj):
    """ Solve dynamic programming problem by backward induction.
    """
    # Antibugging
    assert (robupy_obj.get_status())

    # Distribute class attributes
    init_dict = robupy_obj.get_attr('init_dict')

    num_periods = robupy_obj.get_attr('num_periods')

    num_draws = robupy_obj.get_attr('num_draws')

    edu_start = robupy_obj.get_attr('edu_start')

    ambiguity = robupy_obj.get_attr('ambiguity')

    edu_max = robupy_obj.get_attr('edu_max')

    debug = robupy_obj.get_attr('debug')

    delta = robupy_obj.get_attr('delta')

    seed = robupy_obj.get_attr('seed')

    # Construct auxiliary objects
    level = ambiguity['level']

    with_ambiguity = (level != 0.00)

    if os.path.exists('ambiguity.robupy.log'):
        os.remove('ambiguity.robupy.log')

    if debug and with_ambiguity:
        open('ambiguity.robupy.log', 'w').close()

    # Create grid of admissible state space values.
    states_all, states_number_period, mapping_state_idx = \
        _create_state_space(robupy_obj)

    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Run checks of the state space variables
    if debug is True:
        checks('state_space', robupy_obj, states_all, states_number_period,
                mapping_state_idx)

    # Draw random variables. Handle special case of zero variances as this
    # case is useful for hand-based testing.
    eps_baseline_periods, eps_standard_periods, true_cholesky = \
        _create_eps(seed, num_periods, num_draws, init_dict)

    # Collect special arguments for ambiguity. This setup allows to
    # integrate the interfaces to the risk and ambiguity functions.
    ambiguity_args = None

    if with_ambiguity:
        ambiguity_args = dict()
        ambiguity_args['cholesky'] = true_cholesky
        ambiguity_args['ambiguity'] = ambiguity

    # Select interface to simulate the EMAX appropriately.
    eps_relevant_periods = eps_baseline_periods
    simulate_emax = simulate_emax_risk

    if with_ambiguity:
        eps_relevant_periods = eps_standard_periods
        simulate_emax = simulate_emax_ambiguity

    # Initialize container for expected future values and payoffs. The ex post
    # payoffs and the future payoffs are only stored for debugging reasons
    # and can be removed if memory usage turns out to be an issue. I
    # initialize all infinite values as this allows for easy assertion tests
    # later on.
    emax = np.tile(np.nan, (
        num_periods, max_states_period))
    period_payoffs_ex_post = np.tile(np.nan, (
        num_periods, max_states_period, 4))
    period_future_payoffs = np.tile(np.nan, (
        num_periods, max_states_period, 4))

    # Calculate ex ante payoffs. These are calculated without any reference
    # to the alternative shock distributions.
    logger.info('Staring calculation of ex ante payoffs')

    # Construct coefficients
    coeffs_A = [init_dict['A']['int']] + init_dict['A']['coeff']
    coeffs_B = [init_dict['B']['int']] + init_dict['B']['coeff']

    coeffs_edu = [init_dict['EDUCATION']['int']] + init_dict['EDUCATION'][
        'coeff']
    coeffs_home = [init_dict['HOME']['int']]

    period_payoffs_ex_ante = perf.calculate_payoffs_ex_ante(num_periods,
                                                            states_number_period,
                                                            states_all,
                                                            edu_start, coeffs_A,
                                                            coeffs_B,
                                                            coeffs_edu,
                                                            coeffs_home,
                                                            max_states_period)

    # Logging
    logger.info('... finished \n')

    logger.info('Staring backward induction procedure')

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Logging.
        logger.info('... solving period ' + str(period))

        # Extract disturbances
        eps_relevant = eps_relevant_periods[period, :, :]

        # Loop over all possible states
        for k in range(states_number_period[period]):
            # Extract payoffs
            payoffs_ex_ante = period_payoffs_ex_ante[period, k, :]

            # Simulate the expected future value.
            emax_simulated, payoffs_ex_post, future_payoffs = \
                simulate_emax(num_draws, eps_relevant, period, k,
                              payoffs_ex_ante, edu_max, edu_start,
                              mapping_state_idx,
                              states_all, num_periods, emax, delta, debug,
                              ambiguity_args)

            # Collect information
            period_payoffs_ex_post[period, k, :] = payoffs_ex_post
            period_future_payoffs[period, k, :] = future_payoffs

            # Collect
            emax[period, k] = emax_simulated

    # Logging
    logger.info('... finished \n')

    # Run checks on expected future values and its ingredients
    if debug is True:
        checks('emax', robupy_obj, states_all, states_number_period, emax,
                period_future_payoffs)

    # Summarize optimizations in case of ambiguity.
    if debug is True and with_ambiguity:
        _summarize_ambiguity(num_periods)

    # Update
    robupy_obj.unlock()

    robupy_obj.set_attr('states_number_period', states_number_period)

    robupy_obj.set_attr('period_payoffs_ex_ante', period_payoffs_ex_ante)

    robupy_obj.set_attr('period_payoffs_ex_post', period_payoffs_ex_post)

    robupy_obj.set_attr('period_future_payoffs', period_future_payoffs)

    robupy_obj.set_attr('mapping_state_idx', mapping_state_idx)

    robupy_obj.set_attr('states_all', states_all)

    robupy_obj.set_attr('emax', emax)

    # Set flag that object includes the solution objects.
    robupy_obj.set_attr('is_solved', True)

    # Lock up again
    robupy_obj.lock()

    # Finishing
    return robupy_obj


''' Private functions
'''


def _create_eps(seed, num_periods, num_draws, init_dict):
    """ Create disturbances.
    """
    # Ensure recomputability
    np.random.seed(seed)

    eps_standard_periods = np.random.multivariate_normal(np.zeros(4),
                                                         np.identity(4), (
                                                             num_periods,
                                                             num_draws))

    # Check for special case
    all_zeros = (np.count_nonzero(init_dict['SHOCKS']) == 0)

    # Prepare Cholesky decomposition
    if all_zeros:
        true_cholesky = np.zeros((4, 4))
    else:
        true_cholesky = np.linalg.cholesky(init_dict['SHOCKS'])

    # Construct baseline disturbances
    eps_baseline_periods = np.tile(np.nan, (num_periods, num_draws, 4))

    for period in range(num_periods):
        eps_baseline_periods[period, :, :] = \
            np.dot(true_cholesky, eps_standard_periods[period, :, :].T).T

    # Finishing
    return eps_baseline_periods, eps_standard_periods, true_cholesky


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


def _summarize_ambiguity(num_periods):
    """ Summarize optimizations in case of ambiguity.
    """

    def _process_cases(list_):
        """ Process cases and determine whether keyword or empty line.
        """
        # Antibugging
        assert (isinstance(list_, list))

        # Get information
        is_empty = (len(list_) == 0)

        if not is_empty:
            is_block = list_[0].isupper()
        else:
            is_block = False

        # Antibugging
        assert (is_block in [True, False])
        assert (is_empty in [True, False])

        # Finishing
        return is_empty, is_block

    dict_ = dict()

    for line in open('ambiguity.robupy.log').readlines():

        # Split line
        list_ = shlex.split(line)

        # Determine special cases
        is_empty, is_block = _process_cases(list_)

        # Applicability
        if is_empty:
            continue

        # Prepare dictionary
        if is_block:

            period = int(list_[1])

            if period in dict_.keys():
                continue

            dict_[period] = {}
            dict_[period]['success'] = 0
            dict_[period]['failure'] = 0

        # Collect success indicator
        if list_[0] == 'Success':
            is_success = (list_[1] == 'True')
            if is_success:
                dict_[period]['success'] += 1
            else:
                dict_[period]['failure'] += 1

    with open('ambiguity.robupy.log', 'a') as file_:

        file_.write('SUMMARY\n\n')

        string = '''{0[0]:>10} {0[1]:>10} {0[2]:>10} {0[3]:>10}\n'''

        file_.write(string.format(['Period', 'Total', 'Success', 'Failure']))

        file_.write('\n')

        for period in range(num_periods):
            success = dict_[period]['success']
            failure = dict_[period]['failure']
            total = success + failure

            file_.write(string.format([period, total, success, failure]))
