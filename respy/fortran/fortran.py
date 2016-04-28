""" This module serves as the interface between the PYTHON code and the
FORTRAN implementations.
"""
# standard library
import os

# project library
from respy.fortran.fortran_auxiliary import write_resfort_initialization
from respy.fortran.fortran_auxiliary import write_dataset
from respy.fortran.fortran_auxiliary import get_results
from respy.fortran.fortran_auxiliary import read_data

from respy.python.shared.shared_constants import FORTRAN_DIR


def fort_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta, data_array,
        num_agents_est, num_draws_prob, tau, seed_emax, seed_prob):
    """ This function serves as the interface to the FORTRAN implementations.
    """
    # Prepare RESFORT execution by collecting arguments and writing them to
    # the RESFORT initialization file.
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta)

    args = args + (num_draws_prob, num_agents_est, seed_prob, seed_emax,
        tau, 'evaluate')
    write_resfort_initialization(*args)

    # If an evaluation is requested, then a specially formatted dataset is
    # written to a scratch file. This eases the reading of the dataset in
    # FORTRAN.
    write_dataset(data_array)

    # Call executable
    os.system('"' + FORTRAN_DIR + '/bin/resfort"')

    crit_val = read_data('eval', 1)[0]

    return crit_val


def fort_solve(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta, seed_emax, tau):
    """ This function serves as the interface to the FORTRAN implementations.
    """
    # Prepare RESFORT execution by collecting arguments and writing them to
    # the RESFORT initialization file. The last arguments are just
    # placeholders as that are not needed for a solution request.
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta)

    args = args + (1, 1, 1, seed_emax, tau, 'solve')
    write_resfort_initialization(*args)

    # Call executable
    os.system('"' + FORTRAN_DIR + '/bin/resfort"')

    # Return arguments depends on the request.
    args = get_results(num_periods, min_idx)

    return args
