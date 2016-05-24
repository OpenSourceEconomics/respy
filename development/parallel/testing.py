#!/usr/bin/env python
""" This module tests the software.

"""
import numpy as np
import os

import sys
sys.path.insert(0, '/home/peisenha/restudToolbox/package/respy')

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.fortran.fortran_auxiliary import write_resfort_initialization
from respy.fortran.fortran_auxiliary import write_dataset

from respy.tests.codes.random_init import generate_init

from respy import RespyCls, simulate
from respy.solve import solve
from respy.process import process
from respy.evaluate import evaluate

def get_resfort_init(request):

    respy_obj = RespyCls('test.respy.ini')

    # Distribute class attributes
    model_paras, num_periods, edu_start, is_debug, edu_max, delta, version, \
        num_draws_emax, seed_emax, is_interpolated, num_points, is_myopic, \
        min_idx, store, tau, num_draws_prob, seed_prob, num_agents_est \
        = dist_class_attributes(
        respy_obj, 'model_paras', 'num_periods', 'edu_start', 'is_debug', 'edu_max',
        'delta', 'version', 'num_draws_emax', 'seed_emax', 'is_interpolated',
        'num_points', 'is_myopic', 'min_idx', 'store', 'tau',
        'num_draws_prob', 'seed_prob', 'num_agents_est')

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = dist_model_paras(
        model_paras, is_debug)

    # Collect baseline arguments. These are latter amended to account for
    # each interface.
    if request == 'solve':
        args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated,
        num_draws_emax, num_periods, num_points, is_myopic, edu_start, is_debug,
        edu_max, min_idx, delta)

        args = args + (1, 1, 1, seed_emax, tau, 'solve')
    else:
        args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta)

        args = args + (num_draws_prob, num_agents_est, seed_prob, seed_emax, tau, 'evaluate')


    write_resfort_initialization(*args)

# Test the functionality of the executable for varying number of slaves and
# varying number of model specifications.
np.random.seed(3)
while True:
    # This generates the initialization file

    constr = dict()
    constr['version'] = 'FORTRAN'
    constr['apply'] = False
    generate_init(constr)

    respy_obj = RespyCls('test.respy.ini')
    respy_obj = simulate(respy_obj)
    respy_obj = solve(respy_obj)

    # # This ensures
    request = 'evaluate'
    get_resfort_init(request)

    data_frame = process(respy_obj)
    write_dataset(data_frame.as_matrix())

    base = None
    # # TODO: Just draw a random number of slaves
    num_slaves = np.random.randint(1, 5)

    cmd = 'mpiexec /home/peisenha/restudToolbox/package/respy/fortran/bin' \
          '/resfort_parallel_master ' + str(num_slaves)
    assert os.system(cmd) == 0

    if request == 'solve':
        num_periods = respy_obj.get_attr('num_periods')
        max_states_period = int(np.loadtxt('.max_states_period.resfort.dat'))

        shape = (num_periods, max_states_period)
        periods_emax = np.loadtxt('.periods_emax.resfort.dat')
        periods_emax = np.reshape(periods_emax, shape)
        base = periods_emax[0, 0]
        package = respy_obj.get_attr('periods_emax')[0, 0]
    elif request == 'evaluate':

        package = evaluate(respy_obj)

        write_dataset(data_frame.as_matrix())
        cmd = 'mpiexec /home/peisenha/restudToolbox/package/respy/fortran/bin' \
              '/resfort_parallel_master ' + str(num_slaves)
        assert os.system(cmd) == 0

        base = np.loadtxt('.eval.resfort.dat')


    print('\n\n Visual Debugging')
    print(base, package)
    print('\n\n')

    np.testing.assert_allclose(base, package)

    ###########################################################################

    # Testing parallel vs scalar functions
    #num_slaves = np.random.randint(1, 5)
    #cmd = 'mpiexec /home/peisenha/restudToolbox/package/respy/fortran/bin' \
    #      '/testing_parallel_scalar ' + str(num_slaves)
    #os.system(cmd)
    #assert not os.path.exists('.error.testing')
