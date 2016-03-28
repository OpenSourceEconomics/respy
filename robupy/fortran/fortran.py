""" This module serves as the interface between the PYTHON code and the
FORTRAN implementations.
"""
# standard library
import os

# project library
from robupy.fortran.fortran_auxiliary import write_robufort_initialization
from robupy.fortran.fortran_auxiliary import write_dataset
from robupy.fortran.fortran_auxiliary import get_results
from robupy.fortran.fortran_auxiliary import read_data

from robupy.shared.constants import FORTRAN_DIR

''' Main function
'''

def fort_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug,
        measure, edu_max, min_idx, delta, level, data_array, num_agents,
        num_draws_prob,  seed_emax, seed_prob):
    """ This function serves as the interface to the FORTRAN implementations.
    """

    #base_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
    #    is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
    #    num_periods, num_points, is_myopic, edu_start, is_debug, measure,
    #    edu_max, min_idx, delta, level)

    # Prepare ROBUFORT execution by collecting arguments and writing them to
    # the ROBUFORT initialization file.
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, is_debug,
            min_idx, measure, edu_max, delta, level,
            num_draws_prob, num_agents, seed_prob, seed_emax, 'evaluate')

    write_robufort_initialization(*args)

    # If an evaluation is requested, then a specially formatted dataset is
    # written to a scratch file. This eases the reading of the dataset in
    # FORTRAN.
    write_dataset(data_array)

    # Call executable
    os.system('"' + FORTRAN_DIR + '/bin/robufort"')

    crit_val = read_data('eval', 1)[0]

    return crit_val

def fort_solve(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug,
        measure, edu_max, min_idx, delta, level, seed_emax):
    """ This function serves as the interface to the FORTRAN implementations.
    """
    # Prepare ROBUFORT execution by collecting arguments and writing them to
    # the ROBUFORT initialization file.
    # TODO: ORder the arguments in write_robupfort so that I can do the
    # following args + (1, 1, 1,, solve).

    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, is_debug, min_idx,
            measure, edu_max, delta, level,
            1, 1, 1, seed_emax, 'solve')

    write_robufort_initialization(*args)

    # Call executable
    os.system('"' + FORTRAN_DIR + '/bin/robufort"')

    # Return arguments depends on the request.
    args = get_results(num_periods, min_idx)

    return args