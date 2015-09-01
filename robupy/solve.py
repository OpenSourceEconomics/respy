""" This module contains all the capabilities to solve the dynamic
programming problem.


    This module uses features as object oriented programming. The backend is
    written with a performance based

"""

# standard library
import numpy as np
import logging
import shlex
import os

# project library

from robupy.checks.checks_solve import checks_solve

import robupy.performance.fortran.fortran_functions as fort
import robupy.performance.python.python_functions as py

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')

''' Public function
'''


def solve(robupy_obj):
    """ Solve dynamic programming problem by backward induction.
    """
    # Antibugging
    assert (robupy_obj.get_status())

    # Distribute class attributes
    ambiguity = robupy_obj.get_attr('ambiguity')

    debug = robupy_obj.get_attr('debug')

    # Construct auxiliary objects
    level = ambiguity['level']
    measure = ambiguity['measure']

    with_ambiguity = (level != 0.00)

    if os.path.exists('ambiguity.robupy.log'):
        os.remove('ambiguity.robupy.log')

    if debug and with_ambiguity:
        open('ambiguity.robupy.log', 'w').close()

    # Creating the state space of the model and collect the results in the
    # package class.
    logger.info('Starting state space creation')

    states_all, states_number_period, mapping_state_idx = \
        _wrapper_create_state_space(robupy_obj)

    logger.info('... finished \n')

    robupy_obj.unlock()

    robupy_obj.set_attr('states_number_period', states_number_period)

    robupy_obj.set_attr('mapping_state_idx', mapping_state_idx)

    robupy_obj.set_attr('states_all', states_all)

    robupy_obj.lock()

    # Draw random variables.
    eps_baseline_periods, eps_standard_periods, true_cholesky = \
        _create_eps(robupy_obj)

    if with_ambiguity:
        eps_relevant_periods = eps_standard_periods
    else:
        eps_relevant_periods = eps_baseline_periods

    # Calculate ex ante payoffs which are later used in the backward
    # induction procedure. These are calculated without any reference
    # to the alternative shock distributions.
    logger.info('Starting calculation of ex ante payoffs')

    periods_payoffs_ex_ante = _wrapper_calculate_payoffs_ex_ante(robupy_obj)

    logger.info('... finished \n')

    robupy_obj.unlock()

    robupy_obj.set_attr('period_payoffs_ex_ante', periods_payoffs_ex_ante)

    robupy_obj.lock()

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available.
    logger.info('Staring backward induction procedure')

    periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
        _wrapper_backward_induction_procedure(robupy_obj, eps_relevant_periods,
            true_cholesky, level, measure)

    logger.info('... finished \n')

    # Summarize optimizations in case of ambiguity.
    if debug and with_ambiguity:
        _summarize_ambiguity(robupy_obj)

    # Update
    # TODO: renaming periods?
    robupy_obj.unlock()

    robupy_obj.set_attr('period_payoffs_ex_post', periods_payoffs_ex_post)

    robupy_obj.set_attr('period_future_payoffs', periods_future_payoffs)

    robupy_obj.set_attr('emax', periods_emax)

    # Set flag that object includes the solution objects.
    robupy_obj.set_attr('is_solved', True)

    # Lock up again
    robupy_obj.lock()

    # Finishing
    return robupy_obj


''' Wrappers for core functions
'''


def _wrapper_calculate_payoffs_ex_ante(robupy_obj):
    """ Calculate the ex ante payoffs.
    """
    # Distribute class attributes
    states_number_period = robupy_obj.get_attr('states_number_period')

    num_periods = robupy_obj.get_attr('num_periods')

    states_all = robupy_obj.get_attr('states_all')

    init_dict = robupy_obj.get_attr('init_dict')

    edu_start = robupy_obj.get_attr('edu_start')

    fast = robupy_obj.get_attr('fast')

    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Construct coefficients
    coeffs_a = [init_dict['A']['int']] + init_dict['A']['coeff']
    coeffs_b = [init_dict['B']['int']] + init_dict['B']['coeff']

    coeffs_edu = [init_dict['EDUCATION']['int']] + init_dict['EDUCATION']['coeff']
    coeffs_home = [init_dict['HOME']['int']]

    # Interface to core functions
    if fast:
        periods_payoffs_ex_ante = fort.calculate_payoffs_ex_ante(num_periods,
            states_number_period, states_all, edu_start, coeffs_a, coeffs_b,
            coeffs_edu, coeffs_home, max_states_period)
    else:
        periods_payoffs_ex_ante = py.calculate_payoffs_ex_ante(num_periods,
            states_number_period, states_all, edu_start, coeffs_a, coeffs_b,
            coeffs_edu, coeffs_home, max_states_period)

    # Finishing
    return periods_payoffs_ex_ante


def _wrapper_create_state_space(robupy_obj):
    """ Create state space. This function is a wrapper around the PYTHON and
    FORTRAN implementation.
    """

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    debug = robupy_obj.get_attr('debug')

    fast = robupy_obj.get_attr('fast')

    # Create grid of admissible state space values
    min_idx = min(num_periods, (edu_max - edu_start + 1))

    # Interface to core functions
    if fast:
        states_all, states_number_period, mapping_state_idx = \
            fort.create_state_space(num_periods, edu_start, edu_max, min_idx)
    else:
        states_all, states_number_period, mapping_state_idx = \
            py.create_state_space(num_periods, edu_start, edu_max, min_idx)

    # Type transformations
    states_number_period = np.array(states_number_period, dtype='int')

    # Cutting to size
    states_all = states_all[:, :max(states_number_period), :]

    # Set missing values to NAN
    states_all = _replace_missing_values(states_all)

    mapping_state_idx = _replace_missing_values(mapping_state_idx)

    # Run checks of the state space variables
    if debug:
        checks_solve('_wrapper_create_state_space_out', robupy_obj, states_all,
            states_number_period, mapping_state_idx)

    # Finishing
    return states_all, states_number_period, mapping_state_idx


def _wrapper_backward_induction_procedure(robupy_obj, eps_relevant_periods,
        true_cholesky, level, measure):
    """ Wrapper for backward induction procedure.
    """
    # Distribute class attributes
    period_payoffs_ex_ante = robupy_obj.get_attr('period_payoffs_ex_ante')

    states_number_period = robupy_obj.get_attr('states_number_period')

    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

    num_periods = robupy_obj.get_attr('num_periods')

    states_all = robupy_obj.get_attr('states_all')

    num_draws = robupy_obj.get_attr('num_draws')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    delta = robupy_obj.get_attr('delta')

    debug = robupy_obj.get_attr('debug')

    fast = robupy_obj.get_attr('fast')

    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Interface to core functions
    if fast:
        periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
            fort.backward_induction(num_periods, max_states_period,
                eps_relevant_periods, num_draws, states_number_period,
                period_payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
                states_all, delta, debug, true_cholesky, level, measure)
    else:
        periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
            py.backward_induction(num_periods, max_states_period,
                eps_relevant_periods, num_draws, states_number_period,
                period_payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
                states_all, delta, debug, true_cholesky, level, measure)

    # Set missing values to NAN
    periods_emax = _replace_missing_values(periods_emax)

    periods_future_payoffs = _replace_missing_values(periods_future_payoffs)

    periods_payoffs_ex_post = _replace_missing_values(periods_payoffs_ex_post)

    # Run checks on expected future values and its ingredients
    if debug:
        checks_solve('emax', robupy_obj, periods_emax, periods_future_payoffs)

    # Finishing
    return periods_emax, periods_payoffs_ex_post, periods_future_payoffs


''' Auxiliary functions
'''

def _replace_missing_values(argument):
    """ Replace missing value -99 with NAN
    """
    # Determine missing values
    is_missing = (argument == -99)

    # Transform to float array
    mapping_state_idx = np.asfarray(argument)

    # Replace missing values
    mapping_state_idx[is_missing] = np.nan

    # Finishing
    return mapping_state_idx


def _create_eps(robupy_obj):
    """ Create disturbances.  Handle special case of zero variances as this
        case is useful for hand-based testing.
    """

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    num_draws = robupy_obj.get_attr('num_draws')

    init_dict = robupy_obj.get_attr('init_dict')

    seed = robupy_obj.get_attr('seed_solution')

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


def _summarize_ambiguity(robupy_obj):
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

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

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
