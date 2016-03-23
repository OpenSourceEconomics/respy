""" This module provides the interface to the functionality needed to solve the
model with PYTHON and F2PY capabilities.
"""

# standard library
import logging

import numpy as np
# project library
from robupy.auxiliary import replace_missing_values

import robupy.python.py.python_library as python_library

from robupy.shared.constants import MISSING_FLOAT

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')

''' Main function
'''


def solve_python(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        shocks_cholesky, is_deterministic, is_interpolated, num_draws_emax,
        periods_draws_emax, is_ambiguous, num_periods, num_points, edu_start,
        is_myopic, is_debug, measure, edu_max, min_idx, delta, level,
        is_python):
    """ Solving the model using PYTHON/F2PY code. This purpose of this
    wrapper is to extract all relevant information from the project class to
    pass it on to the actual solution functions. This is required to align
    the functions across the PYTHON and F2PY implementations.
    """
    # Solve the model using PYTHON/F2PY implementation
    # Creating the state space of the model and collect the results in the
    # package class.
    logger.info('Starting state space creation')

    states_all, states_number_period, mapping_state_idx, max_states_period = \
        _create_state_space(num_periods, edu_start, edu_max, min_idx, is_python)

    logger.info('... finished \n')

    # Calculate systematic payoffs which are later used in the backward
    # induction procedure. These are calculated without any reference
    # to the alternative shock distributions.
    logger.info('Starting calculation of systematic payoffs')

    periods_payoffs_systematic = \
        _calculate_payoffs_systematic(coeffs_a, coeffs_b, coeffs_edu,
            coeffs_home, states_number_period, num_periods, states_all,
            edu_start, max_states_period, is_python)

    logger.info('... finished \n')

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available. If agents are myopic, the backward induction
    # procedure is not called upon.
    logger.info('Starting backward induction procedure')

    # Initialize containers, which contain a lot of missing values as we
    # capture the tree structure in arrays of fixed dimension.
    i, j = num_periods, max_states_period
    periods_emax = np.tile(MISSING_FLOAT, (i, j))
    periods_payoffs_ex_post = np.tile(MISSING_FLOAT, (i, j, 4))
    periods_payoffs_future = np.tile(MISSING_FLOAT, (i, j, 4))

    if is_myopic:
        # All other objects remain set to MISSING_FLOAT. This align the
        # treatment for the two special cases: (1) is_myopic and (2)
        # is_interpolated.
        for period, num_states in enumerate(states_number_period):
            periods_emax[period, :num_states] = 0.0

    else:
        periods_emax, periods_payoffs_ex_post, periods_payoffs_future = \
            _backward_induction_procedure(periods_payoffs_systematic,
                states_number_period, mapping_state_idx, is_interpolated,
                num_periods, num_points, states_all, num_draws_emax, edu_start,
                is_debug, edu_max, measure, shocks_cov, delta, level,
                is_ambiguous, periods_draws_emax, is_deterministic,
                max_states_period, shocks_cholesky, is_python)

    # Replace missing values
    periods_emax = replace_missing_values(periods_emax)

    periods_payoffs_future = replace_missing_values(periods_payoffs_future)

    periods_payoffs_ex_post = replace_missing_values(periods_payoffs_ex_post)

    logger.info('... finished \n')

    # Update class attributes with solution
    args = [periods_payoffs_systematic, periods_payoffs_ex_post,
            periods_payoffs_future, states_number_period, mapping_state_idx,
            periods_emax, states_all]

    # Finishing
    return args

''' Auxiliary functions
'''


def _create_state_space(num_periods, edu_start, edu_max, min_idx, is_python):
    """ Create state space. This function is a wrapper around the PYTHON and
    F2PY implementation.
    """
    # Interface to core functions
    if is_python:
        create_state_space = python_library.create_state_space
    else:
        import robupy.fortran.f2py_library as f2py_library
        create_state_space = f2py_library.wrapper_create_state_space

    # Create state space
    states_all, states_number_period, mapping_state_idx, max_states_period = \
        create_state_space(num_periods, edu_start, edu_max, min_idx)

    # Type transformations
    states_number_period = np.array(states_number_period, dtype='int')

    max_states_period = max(states_number_period)

    # Cutting to size
    states_all = states_all[:, :max(states_number_period), :]

    # Set missing values to NAN
    states_all = replace_missing_values(states_all)

    mapping_state_idx = replace_missing_values(mapping_state_idx)

    # Collect arguments
    args = [states_all, states_number_period, mapping_state_idx]
    args += [max_states_period]

    # Finishing
    return args


def _calculate_payoffs_systematic(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        states_number_period, num_periods, states_all, edu_start,
        max_states_period, is_python):
    """ Calculate the systematic payoffs. This function is a wrapper around the
    PYTHON and F2PY implementation.
    """
    # Interface to core functions
    if is_python:
        calculate_payoffs_systematic = \
            python_library.calculate_payoffs_systematic
    else:
        import robupy.fortran.f2py_library as f2py_library
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
        measure, shocks_cov, delta, level, is_ambiguous,
        periods_disturbances_emax, is_deterministic, max_states_period,
        shocks_cholesky, is_python):
    """ Perform backward induction procedure. This function is a wrapper
    around the PYTHON and F2PY implementation.
    """
    # Antibugging
    assert checks('_backward_induction_procedure', delta)

    # Interface to core functions
    if is_python:
        backward_induction = python_library.backward_induction
    else:
        import robupy.fortran.f2py_library as f2py_library
        backward_induction = f2py_library.wrapper_backward_induction

    # Perform backward induction procedure
    periods_emax, periods_payoffs_ex_post, periods_payoffs_future = \
        backward_induction(num_periods, max_states_period,
            periods_disturbances_emax, num_draws_emax, states_number_period,
            periods_payoffs_systematic, edu_max, edu_start,
            mapping_state_idx, states_all, delta, is_debug, shocks_cov, level,
            is_ambiguous, measure, is_interpolated, num_points,
            is_deterministic, shocks_cholesky)

    # Finishing
    return periods_emax, periods_payoffs_ex_post, periods_payoffs_future


def checks(str_, *args):
    """ Some guards to the interfaces.
    """
    if str_ == '_backward_induction_procedure':

        # Distribute input parameters
        delta, = args

        # The backward induction procedure does not work properly for the
        # myopic case anymore. This is necessary as in the special
        # case where delta is equal to zero, (-np.inf * 0.00) evaluates to
        # NAN. This is returned as the maximum value when calling np.argmax.
        # This was preciously handled by an auxiliary function
        # "_stabilize_myopic" inside "get_total_value".
        assert (delta > 0)

    else:

        raise AssertionError

    # Finishing
    return True
