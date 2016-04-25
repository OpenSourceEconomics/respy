""" This module provides the interface to the functionality needed to solve the
model with PYTHON and F2PY capabilities.
"""

# standard library
import logging

import numpy as np

# project library
from respy.python.shared.shared_constants import MISSING_FLOAT

from respy.python.solve.solve_auxiliary import pyth_calculate_payoffs_systematic
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.solve.solve_auxiliary import pyth_backward_induction

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')

''' Main function
'''


def pyth_solve(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug,
        edu_max, min_idx, delta, level, periods_draws_emax):
    """ Solving the model using pure PYTHON code.
    """
    # Construct Cholesky decomposition
    if is_deterministic:
        shocks_cholesky = np.zeros((4, 4))
    else:
        shocks_cholesky = np.linalg.cholesky(shocks_cov)

    # Creating the state space of the model and collect the results in the
    # package class.
    logger.info('Starting state space creation')

    # Create state space
    states_all, states_number_period, mapping_state_idx, max_states_period = \
        pyth_create_state_space(num_periods, edu_start, edu_max, min_idx)

    # Cutting to size
    states_all = states_all[:, :max(states_number_period), :]

    logger.info('... finished \n')

    # Calculate systematic payoffs which are later used in the backward
    # induction procedure. These are calculated without any reference
    # to the alternative shock distributions.
    logger.info('Starting calculation of systematic payoffs')

    # Calculate all systematic payoffs
    periods_payoffs_systematic = pyth_calculate_payoffs_systematic(num_periods,
        states_number_period, states_all, edu_start, coeffs_a, coeffs_b,
        coeffs_edu, coeffs_home, max_states_period)

    logger.info('... finished \n')

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available. If agents are myopic, the backward induction
    # procedure is not called upon.
    logger.info('Starting backward induction procedure')

    # Initialize containers, which contain a lot of missing values as we
    # capture the tree structure in arrays of fixed dimension.
    i, j = num_periods, max_states_period
    periods_emax = np.tile(MISSING_FLOAT, (i, j))

    if is_myopic:
        # All other objects remain set to MISSING_FLOAT. This align the
        # treatment for the two special cases: (1) is_myopic and (2)
        # is_interpolated.
        for period, num_states in enumerate(states_number_period):
            periods_emax[period, :num_states] = 0.0

    else:
        periods_emax = pyth_backward_induction(num_periods, max_states_period,
            periods_draws_emax, num_draws_emax, states_number_period,
            periods_payoffs_systematic, edu_max, edu_start,
            mapping_state_idx, states_all, delta, is_debug, shocks_cov,
            level, is_ambiguous, is_interpolated, num_points,
            is_deterministic, shocks_cholesky)

    logger.info('... finished \n')

    # Collect return arguments in tuple
    args = (periods_payoffs_systematic, states_number_period,
        mapping_state_idx, periods_emax, states_all)

    # Finishing
    return args
