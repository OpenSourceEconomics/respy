""" This module serves as the interface between the PYTHON code and the
FORTRAN implementations.
"""
# standard library
import numpy as np
import pandas as pd

import subprocess
import os

# project library
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_constants import OPTIMIZERS_FORT
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import EXEC_DIR


def resfort_interface(respy_obj, request, data_array=None):
    """ This function provides the interface to the FORTRAN functionality.
    """
    # Add mock specification for FORTRAN optimizers if not defined by user.
    # This is required so the initialization file for FORTRAN is complete.
    respy_obj = add_optimizers(respy_obj)

    # Distribute class attributes
    model_paras, num_periods, edu_start, is_debug, edu_max, delta, \
        num_draws_emax, seed_emax, is_interpolated, num_points_interp, \
        is_myopic, min_idx, tau, is_parallel, num_procs, \
        num_agents_sim, num_draws_prob, num_agents_est, seed_prob, seed_sim, \
        paras_fixed, optimizer_options, optimizer_used, maxfun, paras_fixed = \
            dist_class_attributes(respy_obj,
                'model_paras', 'num_periods', 'edu_start', 'is_debug',
                'edu_max', 'delta', 'num_draws_emax', 'seed_emax',
                'is_interpolated', 'num_points_interp', 'is_myopic', 'min_idx',
                'tau', 'is_parallel', 'num_procs', 'num_agents_sim',
                'num_draws_prob', 'num_agents_est', 'seed_prob', 'seed_sim',
                'paras_fixed', 'optimizer_options', 'optimizer_used', 'maxfun', 'paras_fixed')

    if request == 'estimate':
        # Check that selected optimizer is in line with version of program.
        if maxfun > 0:
            assert optimizer_used in OPTIMIZERS_FORT

        assert data_array is not None
        # If an evaluation is requested, then a specially formatted dataset is
        # written to a scratch file. This eases the reading of the dataset in
        # FORTRAN.
        write_dataset(data_array)

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta)

    args = args + (num_draws_prob, num_agents_est, num_agents_sim, seed_prob,
    seed_emax, tau, num_procs, request, seed_sim, optimizer_options,
    optimizer_used, maxfun, paras_fixed)

    write_resfort_initialization(*args)

    # Call executable
    if not is_parallel:
        cmd = [EXEC_DIR + '/resfort_scalar']
        subprocess.check_call(cmd)
    else:
        cmd = ['mpiexec', EXEC_DIR + '/resfort_parallel_master']
        subprocess.check_call(cmd)

    # Return arguments depends on the request.
    if request == 'solve':
        results = get_results(num_periods, min_idx, num_agents_sim, 'solve')
        args = results[:-1]
    elif request == 'simulate':
        results = get_results(num_periods, min_idx, num_agents_sim, 'simulate')
        args = (results[:-1], results[-1])
    elif request == 'estimate':
        args = None

    return args


def add_optimizers(respy_obj):
    """ This function fills up missing information about optimizers to ensure a
    common interface.
    """
    optimizer_options = respy_obj.get_attr('optimizer_options')


    for optimizer in ['FORT-NEWUOA', 'FORT-BFGS']:

        # Skip if defined by user.
        if optimizer in optimizer_options.keys():
            continue

        if optimizer in ['FORT-NEWUOA']:
            optimizer_options[optimizer] = dict()
            optimizer_options[optimizer]['npt'] = 40
            optimizer_options[optimizer]['rhobeg'] = 0.1
            optimizer_options[optimizer]['rhoend'] = 0.0001
            optimizer_options[optimizer]['maxfun'] = 20

        if optimizer in ['FORT-BFGS']:
            optimizer_options[optimizer] = dict()
            optimizer_options[optimizer]['epsilon'] = 0.00001
            optimizer_options[optimizer]['gtol'] = 0.00001
            optimizer_options[optimizer]['maxiter'] = 10
            optimizer_options[optimizer]['stpmx'] = 100.0

    respy_obj.unlock()
    respy_obj.set_attr('optimizer_options', optimizer_options)
    respy_obj.lock()

    return respy_obj

def get_results(num_periods, min_idx, num_agents_sim, which):
    """ Add results to container.
    """
    # Get the maximum number of states. The special treatment is required as
    # it informs about the dimensions of some of the arrays that are
    # processed below.
    max_states_period = int(np.loadtxt('.max_states_period.resfort.dat'))

    os.unlink('.max_states_period.resfort.dat')

    shape = (num_periods, num_periods, num_periods, min_idx, 2)
    mapping_state_idx = read_data('mapping_state_idx', shape).astype('int')

    shape = (num_periods,)
    states_number_period = \
        read_data('states_number_period', shape).astype('int')

    shape = (num_periods, max_states_period, 4)
    states_all = read_data('states_all', shape).astype('int')

    shape = (num_periods, max_states_period, 4)
    periods_payoffs_systematic = read_data('periods_payoffs_systematic', shape)

    shape = (num_periods, max_states_period)
    periods_emax = read_data('periods_emax', shape)

    # In case of  a simulation, we can also process the simulated dataset.
    if which == 'solve':
        data_array = None
    elif which == 'simulate':
        shape = (num_periods * num_agents_sim, 8)
        data_array = read_data('simulated', shape)
    else:
        raise AssertionError

    # Update class attributes with solution
    args = (periods_payoffs_systematic, states_number_period,
        mapping_state_idx, periods_emax, states_all, data_array)

    # Finishing
    return args


def read_data(label, shape):
    """ Read results
    """
    file_ = '.' + label + '.resfort.dat'

    # This special treatment is required as it is crucial for this data
    # to stay of integer type. All other data is transformed to float in
    # the replacement of missing values.
    if label == 'states_number_period':
        data = np.loadtxt(file_, dtype=np.int64)
    else:
        data = np.loadtxt(file_)

    data = np.reshape(data, shape)

    # Cleanup
    os.unlink(file_)

    # Finishing
    return data


def write_resfort_initialization(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
        num_draws_prob, num_agents_est, num_agents_sim, seed_prob, seed_emax,
        tau, num_procs, request, seed_sim, optimizer_options, optimizer_used, maxfun, paras_fixed):
    """ Write out model request to hidden file .model.resfort.ini.
    """

    # Write out to link file
    with open('.model.resfort.ini', 'w') as file_:

        # BASICS
        line = '{0:10d}\n'.format(num_periods)
        file_.write(line)

        line = '{0:15.10f}\n'.format(delta)
        file_.write(line)

        # WORK
        for num in [coeffs_a, coeffs_b]:
            fmt_ = ' {:15.10f}' * 6 + '\n'
            file_.write(fmt_.format(*num))

        # EDUCATION
        num = coeffs_edu
        line = ' {0:+15.9f} {1:+15.9f} {2:+15.9f}\n'.format(*num)
        file_.write(line)

        line = '{0:10d} '.format(edu_start)
        file_.write(line)

        line = '{0:10d}\n'.format(edu_max)
        file_.write(line)

        # HOME
        line = ' {0:15.10f}\n'.format(coeffs_home[0])
        file_.write(line)

        # SHOCKS
        for j in range(4):
            fmt_ = ' {:20.10f}' * 4 + '\n'
            file_.write(fmt_.format(*shocks_cholesky[j, :]))

        # SOLUTION
        line = '{0:10d}\n'.format(num_draws_emax)
        file_.write(line)

        line = '{0:10d}\n'.format(seed_emax)
        file_.write(line)

        # PROGRAM
        line = '{0}'.format(is_debug)
        file_.write(line + '\n')
        line = '{0:10d}\n'.format(num_procs)
        file_.write(line)

        # INTERPOLATION
        line = '{0}'.format(is_interpolated)
        file_.write(line + '\n')

        line = '{0:10d}\n'.format(num_points_interp)
        file_.write(line)

        # ESTIMATION
        line = '{0:10d}\n'.format(maxfun)
        file_.write(line)

        line = '{0:10d}\n'.format(num_agents_est)
        file_.write(line)

        line = '{0:10d}\n'.format(num_draws_prob)
        file_.write(line)

        line = '{0:10d}\n'.format(seed_prob)
        file_.write(line)

        line = '{0:15.10f}\n'.format(tau)
        file_.write(line)

        # SIMULATION
        line = '{0:10d}\n'.format(num_agents_sim)
        file_.write(line)

        line = '{0:10d}\n'.format(seed_sim)
        file_.write(line)

        # Auxiliary
        line = '{0:10d}\n'.format(min_idx)
        file_.write(line)

        line = '{0}'.format(is_myopic)
        file_.write(line + '\n')

        fmt = '{:} ' * 26
        line = fmt.format(*paras_fixed)
        file_.write(line + '\n')

        # Request
        line = '"{0}"'.format(request)
        file_.write(line + '\n')

        # Directory for executables
        exec_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'

        line = '"{0}"'.format(exec_dir)
        file_.write(line + '\n')

        # Optimizers
        line = '"{0}"\n'.format(optimizer_used)
        file_.write(line)

        line = '{0:10d}\n'.format(optimizer_options['FORT-NEWUOA']['npt'])
        file_.write(line)

        line = '{0:10d}\n'.format(optimizer_options['FORT-NEWUOA']['maxfun'])
        file_.write(line)

        line = ' {0:15.10f}\n'.format(optimizer_options['FORT-NEWUOA']['rhobeg'])
        file_.write(line)

        line = ' {0:15.10f}\n'.format(optimizer_options['FORT-NEWUOA']['rhoend'])
        file_.write(line)

        line = ' {0:15.10f}\n'.format(optimizer_options['FORT-BFGS']['epsilon'])
        file_.write(line)

        line = ' {0:15.10f}\n'.format(optimizer_options['FORT-BFGS']['gtol'])
        file_.write(line)

        line = ' {0:15.10f}\n'.format(optimizer_options['FORT-BFGS']['stpmx'])
        file_.write(line)

        line = '{0:10d}\n'.format(optimizer_options['FORT-BFGS']['maxiter'])
        file_.write(line)


def write_dataset(data_array):
    """ Write the dataset to a temporary file. Missing values are set
    to large values.
    """
    # Transfer to data frame as this allows to fill the missing values with
    # HUGE FLOAT. The numpy array is passed in to align the interfaces across
    # implementations
    data_frame = pd.DataFrame(data_array)
    with open('.data.resfort.dat', 'w') as file_:
        data_frame.to_string(file_, index=False,
            header=None, na_rep=str(HUGE_FLOAT))

    # An empty line is added as otherwise this might lead to problems on the
    # TRAVIS servers. The FORTRAN routine read_dataset() raises an error.
    with open('.data.resfort.dat', 'a') as file_:
        file_.write('\n')