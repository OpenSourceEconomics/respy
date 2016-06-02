""" This module serves as the interface between the PYTHON code and the
FORTRAN implementations.
"""
# standard library
import subprocess

# project library
from respy.fortran.fortran_auxiliary import write_resfort_initialization
from respy.fortran.fortran_auxiliary import write_dataset
from respy.fortran.fortran_auxiliary import get_results
from respy.fortran.fortran_auxiliary import read_data

from respy.python.shared.shared_constants import FORTRAN_DIR


def fort_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta, data_array,
        num_agents_est, num_draws_prob, tau, seed_emax, seed_prob,
        is_parallel, num_procs):
    """ This function serves as the interface to the FORTRAN implementations.
    """
    # Prepare RESFORT execution by collecting arguments and writing them to
    # the RESFORT initialization file.
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta)

    args = args + (num_draws_prob, num_agents_est, seed_prob, seed_emax,
        tau, num_procs, 'evaluate')
    write_resfort_initialization(*args)

    # If an evaluation is requested, then a specially formatted dataset is
    # written to a scratch file. This eases the reading of the dataset in
    # FORTRAN.
    write_dataset(data_array)

    # Call executable
    if not is_parallel:
        cmd = FORTRAN_DIR + '/resfort_scalar'
        subprocess.call(cmd, shell=True)
    else:
        cmd = 'mpiexec ' + FORTRAN_DIR + '/resfort_parallel_master'
        subprocess.call(cmd, shell=True)

    crit_val = read_data('eval', 1)[0]

    return crit_val


def fort_solve(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta, seed_emax, tau,
        is_parallel, num_procs):
    """ This function serves as the interface to the FORTRAN implementations.
    """
    # Prepare RESFORT execution by collecting arguments and writing them to
    # the RESFORT initialization file. The last arguments are just
    # placeholders as that are not needed for a solution request.
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta)

    args = args + (1, 1, 1, seed_emax, tau, num_procs, 'solve')
    write_resfort_initialization(*args)

    # Call executable
    if not is_parallel:
        cmd = FORTRAN_DIR + '/resfort_scalar'
        subprocess.call(cmd, shell=True)
    else:
        cmd = 'mpiexec ' + FORTRAN_DIR + '/resfort_parallel_master'
        subprocess.call(cmd, shell=True)

    # Return arguments depends on the request.
    args = get_results(num_periods, min_idx)

    return args
