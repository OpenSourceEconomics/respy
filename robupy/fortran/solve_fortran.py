""" This module provides the interface to the functionality needed to solve the
model with FORTRAN.
"""
# standard library
import os

# project library
from robupy.fortran.auxiliary import _write_robufort_initialization
from robupy.fortran.auxiliary import _add_results
from robupy.constants.constants import FORTRAN_DIR

''' Main function
'''


def solve_fortran(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, seed_emax,
            is_debug, min_idx, measure, edu_max, delta, level,
            num_draws_prob, num_agents, seed_prob, seed_data, request):
    """ Solve dynamic programming using FORTRAN.
    """

    # Prepare ROBUFORT execution
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, seed_emax,
            is_debug, min_idx, measure, edu_max, delta, level,
            num_draws_prob, num_agents, seed_prob, seed_data, 'solve')

    _write_robufort_initialization(*args)

    # Call executable
    os.system('"' + FORTRAN_DIR + '/bin/robufort"')

    # Add results
    args = _add_results(num_periods, min_idx, 'solve')

    # Finishing
    return args

