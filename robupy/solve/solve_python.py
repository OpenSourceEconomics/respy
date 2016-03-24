""" This module provides the interface to the functionality needed to solve the
model with PYTHON and F2PY capabilities.
"""

# standard library
import logging

import numpy as np

# project library
from robupy.shared.auxiliary import replace_missing_values

from robupy.shared.constants import MISSING_FLOAT

from robupy.solve.solve_auxiliary import pyth_calculate_payoffs_systematic
from robupy.solve.solve_auxiliary import pyth_create_state_space
from robupy.solve.solve_auxiliary import pyth_backward_induction

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')

''' Main function
'''


def pyth_solve(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        shocks_cholesky, is_deterministic, is_interpolated, num_draws_emax,
        periods_draws_emax, is_ambiguous, num_periods, num_points, edu_start,
        is_myopic, is_debug, measure, edu_max, min_idx, delta, level):
    """ Solving the model using pure PYTHON code.
    """
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
            pyth_backward_induction(num_periods, max_states_period,
                periods_draws_emax, num_draws_emax, states_number_period,
                periods_payoffs_systematic, edu_max, edu_start,
                mapping_state_idx, states_all, delta, is_debug, shocks_cov, level,
                is_ambiguous, measure, is_interpolated, num_points,
                is_deterministic, shocks_cholesky)

    logger.info('... finished \n')

    # Update class attributes with solution
    args = [periods_payoffs_systematic, periods_payoffs_ex_post,
            periods_payoffs_future, states_number_period, mapping_state_idx,
            periods_emax, states_all]

    # Finishing
    return args
