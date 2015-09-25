""" This module provides the interface to the functionality needed to solve the
model with Python and F2PY capabilities.
"""

# standard library
import numpy as np
import logging
import os

# project library
from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import create_disturbances

import robupy.python.py.python_library as python_library

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')


def solve_python(robupy_obj):
    """ Solve using PYTHON and F2PY functions
    """
    # Distribute class attributes
    measure = robupy_obj.get_attr('measure')

    shocks = robupy_obj.get_attr('shocks')

    level = robupy_obj.get_attr('level')

    store = robupy_obj.get_attr('store')

    # Construct auxiliary objects
    _start_ambiguity_logging(robupy_obj)

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

    # Get the relevant set of disturbances. These are standard normal draws
    # in the case of an ambiguous world.
    periods_eps_relevant = create_disturbances(robupy_obj, False)

    # Calculate ex ante payoffs which are later used in the backward
    # induction procedure. These are calculated without any reference
    # to the alternative shock distributions.
    logger.info('Starting calculation of ex ante payoffs')

    periods_payoffs_ex_ante = _wrapper_calculate_payoffs_ex_ante(robupy_obj)

    logger.info('... finished \n')

    robupy_obj.unlock()

    robupy_obj.set_attr('periods_payoffs_ex_ante', periods_payoffs_ex_ante)

    robupy_obj.lock()

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available.
    logger.info('Staring backward induction procedure')

    periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
        _wrapper_backward_induction_procedure(robupy_obj, periods_eps_relevant,
            shocks, level, measure)

    logger.info('... finished \n')

    robupy_obj.unlock()

    robupy_obj.set_attr('periods_payoffs_ex_post', periods_payoffs_ex_post)

    robupy_obj.set_attr('periods_future_payoffs', periods_future_payoffs)

    robupy_obj.set_attr('periods_emax', periods_emax)

    robupy_obj.lock()

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

    is_python = robupy_obj.get_attr('is_python')

    init_dict = robupy_obj.get_attr('init_dict')

    edu_start = robupy_obj.get_attr('edu_start')

    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Construct coefficients
    coeffs_a = [init_dict['A']['int']] + init_dict['A']['coeff']
    coeffs_b = [init_dict['B']['int']] + init_dict['B']['coeff']

    coeffs_edu = [init_dict['EDUCATION']['int']] + init_dict['EDUCATION']['coeff']
    coeffs_home = [init_dict['HOME']['int']]

    # Interface to core functions
    if is_python:
        periods_payoffs_ex_ante = python_library.calculate_payoffs_ex_ante(
            num_periods, states_number_period, states_all, edu_start,
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)
    else:
        import robupy.python.f2py.f2py_library as f2py_library
        periods_payoffs_ex_ante = \
            f2py_library.wrapper_calculate_payoffs_ex_ante(num_periods,
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

    is_python = robupy_obj.get_attr('is_python')

    edu_max = robupy_obj.get_attr('edu_max')

    min_idx = robupy_obj.get_attr('min_idx')

    # Interface to core functions
    if is_python:
        states_all, states_number_period, mapping_state_idx = \
            python_library.create_state_space(num_periods, edu_start, edu_max,
                min_idx)
    else:
        import robupy.python.f2py.f2py_library as f2py_library
        states_all, states_number_period, mapping_state_idx = \
            f2py_library.wrapper_create_state_space(num_periods, edu_start,
                edu_max, min_idx)

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
        shocks, level, measure):
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

    is_python = robupy_obj.get_attr('is_python')

    is_debug = robupy_obj.get_attr('is_debug')

    edu_max = robupy_obj.get_attr('edu_max')

    delta = robupy_obj.get_attr('delta')

    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Interface to core functions
    if is_python:
        periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
            python_library.backward_induction(num_periods, max_states_period,
                periods_eps_relevant, num_draws, states_number_period,
                periods_payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
                states_all, delta, is_debug, shocks, level, measure)
    else:
        import robupy.python.f2py.f2py_library as f2py_library
        periods_emax, periods_payoffs_ex_post, periods_future_payoffs = \
            f2py_library.wrapper_backward_induction(num_periods,
                max_states_period, periods_eps_relevant, num_draws,
                states_number_period, periods_payoffs_ex_ante, edu_max,
                edu_start, mapping_state_idx, states_all, delta, is_debug,
                shocks, level, measure)

    # Replace missing values
    periods_emax = replace_missing_values(periods_emax)

    periods_future_payoffs = replace_missing_values(periods_future_payoffs)

    periods_payoffs_ex_post = replace_missing_values(periods_payoffs_ex_post)

    # Finishing
    return periods_emax, periods_payoffs_ex_post, periods_future_payoffs


''' Auxiliary functions
'''


def _start_ambiguity_logging(robupy_obj):
    """ Start logging for ambiguity.
    """
    # Distribute class attributes
    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    is_debug = robupy_obj.get_attr('is_debug')

    # Start logging if required
    if os.path.exists('ambiguity.robupy.log'):
        os.remove('ambiguity.robupy.log')

    if is_debug and is_ambiguous:
        open('ambiguity.robupy.log', 'w').close()

