""" This module provides the interface to the functionality needed to solve the
model with Python and F2PY capabilities.
"""

# standard library
import numpy as np
import logging
import shlex
import os

# project library
from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import read_restud_disturbances

import robupy.python.py.python_core as python_core
try:
    import robupy.python.f2py.f2py_core as f2py_core
except ImportError:
    pass

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')


def solve_python(robupy_obj):
    """ Solve using PYTHON and F2PY functions
    """
    # Distribute class attributes
    measure = robupy_obj.get_attr('measure')

    debug = robupy_obj.get_attr('debug')

    level = robupy_obj.get_attr('level')

    store = robupy_obj.get_attr('store')

    # Construct auxiliary objects
    with_ambiguity = _start_ambiguity_logging(robupy_obj)

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

    # Draw a set of standard normal unobservable disturbances.
    periods_eps_relevant, eps_cholesky = _create_eps(robupy_obj)

    # Calculate ex ante payoffs which are later used in the backward
    # induction procedure. These are calculated without any reference
    # to the alternative shock distributions.
    logger.info('Starting calculation of ex ante payoffs')

    periods_payoffs_ex_ante = _wrapper_calculate_payoffs_ex_ante(robupy_obj)

    logger.info('... finished \n')

    robupy_obj.unlock()

    robupy_obj.set_attr('periods_payoffs_ex_ante', periods_payoffs_ex_ante)

    robupy_obj.set_attr('eps_cholesky', eps_cholesky)

    robupy_obj.lock()

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available.
    logger.info('Staring backward induction procedure')

    periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
        _wrapper_backward_induction_procedure(robupy_obj, periods_eps_relevant,
            eps_cholesky, level, measure)

    logger.info('... finished \n')

    robupy_obj.unlock()

    robupy_obj.set_attr('periods_payoffs_ex_post', periods_payoffs_ex_post)

    robupy_obj.set_attr('periods_future_payoffs', periods_future_payoffs)

    robupy_obj.set_attr('periods_emax', periods_emax)

    robupy_obj.lock()

    # Summarize optimizations in case of ambiguity.
    if debug and with_ambiguity:
        _summarize_ambiguity(robupy_obj)

    # Set flag that object includes the solution objects.
    robupy_obj.unlock()

    robupy_obj.set_attr('is_solved', True)

    robupy_obj.lock()

    # Store object to file
    if store:
        robupy_obj.store('solution.robupy.pkl')

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

    version = robupy_obj.get_attr('version')

    # Auxiliary objects
    max_states_period = max(states_number_period)
    is_f2py = (version == 'F2PY')

    # Construct coefficients
    coeffs_a = [init_dict['A']['int']] + init_dict['A']['coeff']
    coeffs_b = [init_dict['B']['int']] + init_dict['B']['coeff']

    coeffs_edu = [init_dict['EDUCATION']['int']] + init_dict['EDUCATION']['coeff']
    coeffs_home = [init_dict['HOME']['int']]

    # Interface to core functions
    if is_f2py:
        periods_payoffs_ex_ante = \
            f2py_core.wrapper_calculate_payoffs_ex_ante(num_periods,
            states_number_period, states_all, edu_start, coeffs_a, coeffs_b,
            coeffs_edu, coeffs_home, max_states_period)
    else:
        periods_payoffs_ex_ante = python_core.calculate_payoffs_ex_ante(num_periods,
            states_number_period, states_all, edu_start, coeffs_a, coeffs_b,
            coeffs_edu, coeffs_home, max_states_period)

    # Set missing values to NAN
    periods_payoffs_ex_ante = replace_missing_values(periods_payoffs_ex_ante)

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

    version = robupy_obj.get_attr('version')

    min_idx = robupy_obj.get_attr('min_idx')

    debug = robupy_obj.get_attr('debug')

    # Auxiliary objects
    is_f2py = (version == 'F2PY')

    # Interface to core functions
    if is_f2py:
        states_all, states_number_period, mapping_state_idx = \
            f2py_core.wrapper_create_state_space(num_periods, edu_start,
                edu_max, min_idx)
    else:
        states_all, states_number_period, mapping_state_idx = \
            python_core.create_state_space(num_periods, edu_start, edu_max,
                min_idx)

    # Type transformations
    states_number_period = np.array(states_number_period, dtype='int')

    # Cutting to size
    states_all = states_all[:, :max(states_number_period), :]

    # Set missing values to NAN
    states_all = replace_missing_values(states_all)

    mapping_state_idx = replace_missing_values(mapping_state_idx)

    # Finishing
    return states_all, states_number_period, mapping_state_idx


def _wrapper_backward_induction_procedure(robupy_obj, periods_eps_relevant,
        eps_cholesky, level, measure):
    """ Wrapper for backward induction procedure.
    """
    # Distribute class attributes
    periods_payoffs_ex_ante = robupy_obj.get_attr('periods_payoffs_ex_ante')

    states_number_period = robupy_obj.get_attr('states_number_period')

    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

    num_periods = robupy_obj.get_attr('num_periods')

    states_all = robupy_obj.get_attr('states_all')

    num_draws = robupy_obj.get_attr('num_draws')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    delta = robupy_obj.get_attr('delta')

    debug = robupy_obj.get_attr('debug')

    version = robupy_obj.get_attr('version')

    # Auxiliary objects
    max_states_period = max(states_number_period)
    is_f2py = (version == 'F2PY')

    # Interface to core functions
    if is_f2py:
        periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
            f2py_core.wrapper_backward_induction(num_periods,
                max_states_period, periods_eps_relevant, num_draws,
                states_number_period, periods_payoffs_ex_ante, edu_max,
                edu_start, mapping_state_idx,
                states_all, delta)
    else:
        periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
            python_core.backward_induction(num_periods, max_states_period,
                periods_eps_relevant, num_draws, states_number_period,
                periods_payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
                states_all, delta, debug, eps_cholesky, level, measure)

    # Replace missing values
    periods_emax = replace_missing_values(periods_emax)

    periods_future_payoffs = replace_missing_values(periods_future_payoffs)

    periods_payoffs_ex_post = replace_missing_values(periods_payoffs_ex_post)

    # Finishing
    return periods_emax, periods_payoffs_ex_post, periods_future_payoffs


''' Auxiliary functions
'''


def _create_eps(robupy_obj):
    """ Create disturbances.  Handle special case of zero variances as this
    case is useful for hand-based testing. The disturbances are drawn from a
    standard normal distribution and transformed later in the code.
    """
    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    num_draws = robupy_obj.get_attr('num_draws')

    seed = robupy_obj.get_attr('seed_solution')

    shocks = robupy_obj.get_attr('shocks')

    debug = robupy_obj.get_attr('debug')

    level = robupy_obj.get_attr('level')

    # Auxiliary objects
    with_ambiguity = (level > 0.00)

    # Prepare Cholesky decomposition
    all_zeros = (np.count_nonzero(shocks) == 0)
    if all_zeros:
        eps_cholesky = np.zeros((4, 4))
    else:
        eps_cholesky = np.linalg.cholesky(shocks)

    # Draw random disturbances and adjust them for the two occupations
    np.random.seed(seed)
    periods_eps_relevant = np.random.multivariate_normal(np.zeros(4),
        np.identity(4), (num_periods, num_draws))

    for period in range(num_periods):
        periods_eps_relevant[period, :, :] = np.dot(eps_cholesky,
            periods_eps_relevant[period, :, :].T).T
        for j in [0, 1]:
            periods_eps_relevant[period, :, j] = np.exp(periods_eps_relevant[
                                                  period, :, j])

    # This is only used to compare the RESTUD program to the ROBUPY package.
    # It aligns the random components between the two. It is only used in the
    # development process.
    if debug and os.path.isfile('disturbances.txt'):
        periods_eps_relevant = read_restud_disturbances(robupy_obj)

    # In the case of ambiguity, standard normal deviates are passed into the
    # routine. This is unsatisfactory, but required to compare the outputs
    # between the RESTUD program and the ROBUPY package. If standard deviates
    # are passed in from the beginning, the alignment of randomness between
    # the two program yields a too large precision loss.
    if with_ambiguity:
        np.random.seed(seed)
        periods_eps_relevant = np.random.multivariate_normal(np.zeros(4),
            np.identity(4), (num_periods, num_draws))

    # Finishing
    return periods_eps_relevant, eps_cholesky


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


def _start_ambiguity_logging(robupy_obj):
    """ Start logging for ambiguity.
    """
    # Distribute class attributes
    level = robupy_obj.get_attr('level')

    debug = robupy_obj.get_attr('debug')

    # Start logging if required
    with_ambiguity = (level != 0.00)

    if os.path.exists('ambiguity.robupy.log'):
        os.remove('ambiguity.robupy.log')

    if debug and with_ambiguity:
        open('ambiguity.robupy.log', 'w').close()

    # Finishing
    return with_ambiguity