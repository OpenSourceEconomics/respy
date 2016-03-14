""" This module provides the interface to the functionality needed to solve the
model with PYTHON and F2PY capabilities.
"""

# standard library
import numpy as np

import logging
import os

# project library
from robupy.auxiliary import distribute_model_paras
from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import create_disturbances

import robupy.python.py.python_library as python_library

from robupy.constants import MISSING_FLOAT

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')

''' Main function
'''


def solve_python(robupy_obj):
    """ Solving the model using PYTHON/F2PY code. This purpose of this
    wrapper is to extract all relevant information from the project class to
    pass it on to the actual solution functions. This is required to align
    the functions across the PYTHON and F2PY implementations.
    """
    # Distribute class attributes
    is_deterministic = robupy_obj.get_attr('is_deterministic')

    is_interpolated = robupy_obj.get_attr('is_interpolated')

    num_draws_emax = robupy_obj.get_attr('num_draws_emax')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    num_periods = robupy_obj.get_attr('num_periods')

    model_paras = robupy_obj.get_attr('model_paras')

    num_points = robupy_obj.get_attr('num_points')

    seed_emax = robupy_obj.get_attr('seed_emax')

    edu_start = robupy_obj.get_attr('edu_start')

    is_python = robupy_obj.get_attr('is_python')

    is_myopic = robupy_obj.get_attr('is_myopic')

    is_debug = robupy_obj.get_attr('is_debug')

    measure = robupy_obj.get_attr('measure')

    edu_max = robupy_obj.get_attr('edu_max')

    min_idx = robupy_obj.get_attr('min_idx')

    store = robupy_obj.get_attr('store')

    delta = robupy_obj.get_attr('delta')

    level = robupy_obj.get_attr('level')

    # Construct auxiliary objects
    _start_ambiguity_logging(is_ambiguous, is_debug)

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, shocks_cholesky = \
        distribute_model_paras(model_paras, is_debug)

    # Get the relevant set of disturbances. These are standard normal draws
    # in the case of an ambiguous world. This function is located outside the
    # actual bare solution algorithm to ease testing across implementations.
    disturbances_emax = create_disturbances(num_periods, num_draws_emax,
        seed_emax, is_debug, 'emax', shocks_cholesky, is_ambiguous)

    # Solve the model using PYTHON/F2PY implementation
    args = solve_python_bare(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                shocks, edu_max, delta, edu_start, is_debug, is_interpolated,
                level, measure, min_idx, num_draws_emax, num_periods,
                num_points, is_ambiguous, disturbances_emax, is_deterministic,
                is_myopic, is_python)

    # Distribute return arguments
    mapping_state_idx, periods_emax, periods_payoffs_future, \
        periods_payoffs_ex_post, periods_payoffs_systematic, states_all, \
        states_number_period = args

    # Update class attributes with solution
    robupy_obj.unlock()

    robupy_obj.set_attr('periods_payoffs_systematic', periods_payoffs_systematic)

    robupy_obj.set_attr('periods_payoffs_ex_post', periods_payoffs_ex_post)

    robupy_obj.set_attr('periods_payoffs_future', periods_payoffs_future)

    robupy_obj.set_attr('states_number_period', states_number_period)

    robupy_obj.set_attr('mapping_state_idx', mapping_state_idx)

    robupy_obj.set_attr('periods_emax', periods_emax)

    robupy_obj.set_attr('states_all', states_all)

    robupy_obj.set_attr('is_solved', True)

    robupy_obj.lock()

    # Store object to file
    if store:
        robupy_obj.store('solution.robupy.pkl')

    # Finishing
    return robupy_obj

''' Auxiliary functions
'''


def solve_python_bare(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
        edu_max, delta, edu_start, is_debug, is_interpolated, level, measure,
        min_idx, num_draws_emax, num_periods, num_points, is_ambiguous,
        disturbances_emax, is_deterministic, is_myopic, is_python):
    """ This function is required to ensure a full analogy to F2PY and
    FORTRAN implementations. This function is not private to the module as it
    is accessed in the evaluation and optimization modules as well.
    """
    # Creating the state space of the model and collect the results in the
    # package class.
    logger.info('Starting state space creation')

    states_all, states_number_period, mapping_state_idx = \
        _create_state_space(num_periods, edu_start, edu_max, min_idx, is_python)

    logger.info('... finished \n')

    # Calculate systematic payoffs which are later used in the backward
    # induction procedure. These are calculated without any reference
    # to the alternative shock distributions.
    logger.info('Starting calculation of systematic payoffs')

    periods_payoffs_systematic = \
        _calculate_payoffs_systematic(coeffs_a, coeffs_b, coeffs_edu,
            coeffs_home, states_number_period, num_periods, states_all,
            edu_start, is_python)

    logger.info('... finished \n')

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available. If agents are myopic, the backward induction
    # procedure is not called upon.
    logger.info('Starting backward induction procedure')

    if is_myopic:
        # Auxiliary objects
        max_states_period = max(states_number_period)

        i, j = num_periods, max_states_period

        periods_emax = np.tile(MISSING_FLOAT, (i, j))
        periods_payoffs_ex_post = np.tile(MISSING_FLOAT, (i, j, 4))
        periods_payoffs_future = np.tile(MISSING_FLOAT, (i, j, 4))

        # The other objects remain set to missing.
        for period, num_states in enumerate(states_number_period):
            periods_emax[period, :num_states] = 0.0

    else:

        periods_emax, periods_payoffs_ex_post, periods_payoffs_future = \
            _backward_induction_procedure(periods_payoffs_systematic,
                states_number_period, mapping_state_idx, is_interpolated,
                num_periods, num_points, states_all, num_draws_emax, edu_start,
                is_debug, edu_max, measure, shocks, delta, level, is_ambiguous,
                disturbances_emax, is_deterministic, is_python)

    # Replace missing values
    periods_emax = replace_missing_values(periods_emax)

    periods_payoffs_future = replace_missing_values(periods_payoffs_future)

    periods_payoffs_ex_post = replace_missing_values(periods_payoffs_ex_post)

    logger.info('... finished \n')

    # Collect return arguments
    args = [mapping_state_idx, periods_emax, periods_payoffs_future]
    args += [periods_payoffs_ex_post, periods_payoffs_systematic, states_all]
    args += [states_number_period]

    # Finishing
    return args


def _create_state_space(num_periods, edu_start, edu_max, min_idx, is_python):
    """ Create state space. This function is a wrapper around the PYTHON and
    F2PY implementation.
    """
    # Interface to core functions
    if is_python:
        create_state_space = python_library.create_state_space
    else:
        import robupy.python.f2py.f2py_library as f2py_library
        create_state_space = f2py_library.wrapper_create_state_space

    # Create state space
    states_all, states_number_period, mapping_state_idx = \
        create_state_space(num_periods, edu_start, edu_max, min_idx)

    # Type transformations
    states_number_period = np.array(states_number_period, dtype='int')

    # Cutting to size
    states_all = states_all[:, :max(states_number_period), :]

    # Set missing values to NAN
    states_all = replace_missing_values(states_all)

    mapping_state_idx = replace_missing_values(mapping_state_idx)

    # Finishing
    return states_all, states_number_period, mapping_state_idx


def _calculate_payoffs_systematic(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        states_number_period, num_periods, states_all, edu_start, is_python):
    """ Calculate the systematic payoffs. This function is a wrapper around the
    PYTHON and F2PY implementation.
    """
    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Interface to core functions
    if is_python:
        calculate_payoffs_systematic = \
            python_library.calculate_payoffs_systematic
    else:
        import robupy.python.f2py.f2py_library as f2py_library
        calculate_payoffs_systematic = \
            f2py_library.wrapper_calculate_payoffs_systematic

    # Calculate all systematic payoffs
    periods_payoffs_systematic = calculate_payoffs_systematic(num_periods,
        states_number_period, states_all, edu_start, coeffs_a, coeffs_b,
        coeffs_edu, coeffs_home, max_states_period)

    # Set missing values to NAN
    periods_payoffs_systematic = \
        replace_missing_values(periods_payoffs_systematic)

    # Finishing
    return periods_payoffs_systematic


def _backward_induction_procedure(periods_payoffs_systematic,
        states_number_period, mapping_state_idx, is_interpolated, num_periods,
        num_points, states_all, num_draws_emax, edu_start, is_debug, edu_max,
        measure, shocks, delta, level, is_ambiguous, disturbances_emax,
        is_deterministic, is_python):
    """ Perform backward induction procedure. This function is a wrapper
    around the PYTHON and F2PY implementation.
    """
    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Interface to core functions
    if is_python:
        backward_induction = python_library.backward_induction
    else:
        import robupy.python.f2py.f2py_library as f2py_library
        backward_induction = f2py_library.wrapper_backward_induction

    # Perform backward induction procedure
    periods_emax, periods_payoffs_ex_post, periods_payoffs_future = \
        backward_induction(num_periods, max_states_period,
            disturbances_emax, num_draws_emax, states_number_period,
            periods_payoffs_systematic, edu_max, edu_start,
            mapping_state_idx, states_all, delta, is_debug, shocks, level,
            is_ambiguous, measure, is_interpolated, num_points,
            is_deterministic)

    # Finishing
    return periods_emax, periods_payoffs_ex_post, periods_payoffs_future


def _start_ambiguity_logging(is_ambiguous, is_debug):
    """ Start logging for ambiguity.
    """
    # Start logging if required
    if os.path.exists('ambiguity.robupy.log'):
        os.remove('ambiguity.robupy.log')

    if is_debug and is_ambiguous:
        open('ambiguity.robupy.log', 'w').close()
