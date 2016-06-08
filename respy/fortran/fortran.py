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
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras

from respy.python.shared.shared_constants import EXEC_DIR


def resfort_interface(respy_obj, request, data_array=None):

    if request == 'estimate':
        assert data_array is not None
        # If an evaluation is requested, then a specially formatted dataset is
        # written to a scratch file. This eases the reading of the dataset in
        # FORTRAN.
        write_dataset(data_array)

    model_paras, num_periods, edu_start, is_debug, edu_max, delta, \
        version, num_draws_emax, seed_emax, is_interpolated, num_points_interp, \
        is_myopic, min_idx, store, tau, is_parallel, num_procs, \
        num_agents_sim, num_draws_prob, num_agents_est, seed_prob, seed_sim\
        = \
            dist_class_attributes(respy_obj,
                'model_paras', 'num_periods', 'edu_start', 'is_debug',
                'edu_max', 'delta', 'version', 'num_draws_emax', 'seed_emax',
                'is_interpolated', 'num_points_interp', 'is_myopic', 'min_idx',
                'store', 'tau', 'is_parallel', 'num_procs', 'num_agents_sim',
                'num_draws_prob', 'num_agents_est', 'seed_prob', 'seed_sim')

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta)

    args = args + (num_draws_prob, num_agents_est, num_agents_sim, seed_prob,
    seed_emax, tau, num_procs, request, seed_sim)

    write_resfort_initialization(*args)

    # Call executable
    if not is_parallel:
        cmd = EXEC_DIR + '/resfort_scalar'
        subprocess.call(cmd, shell=True)
    else:
        cmd = 'mpiexec ' + EXEC_DIR + '/resfort_parallel_master'
        subprocess.call(cmd, shell=True)

    # Return arguments depends on the request.
    if request == 'solve':
        args = get_results(num_periods, min_idx, num_agents_sim)[:-1]
    elif request == 'estimate':
        args = read_data('eval', 1)[0]
    elif request == 'simulate':
        # TODO: pass abck the solution as well?
        args = get_results(num_periods, min_idx, num_agents_sim)[-1]


    return args
