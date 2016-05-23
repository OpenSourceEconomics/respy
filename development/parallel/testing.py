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


from respy.tests.codes.random_init import generate_init

from respy import RespyCls

def get_random_request():
    generate_init()
    respy_obj = RespyCls('test.respy.ini')
    # Distribute class attributes
    model_paras, num_periods, edu_start, is_debug, edu_max, delta, version, \
        num_draws_emax, seed_emax, is_interpolated, num_points, is_myopic, \
        min_idx, store, tau = dist_class_attributes(
        respy_obj, 'model_paras', 'num_periods', 'edu_start', 'is_debug', 'edu_max',
        'delta', 'version', 'num_draws_emax', 'seed_emax', 'is_interpolated',
        'num_points', 'is_myopic', 'min_idx', 'store', 'tau')

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = dist_model_paras(
        model_paras, is_debug)

    # Collect baseline arguments. These are latter amended to account for
    # each interface.


    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated,
    num_draws_emax, num_periods, num_points, is_myopic, edu_start, is_debug,
    edu_max, min_idx, delta)

    args = args + (1, 1, 1, seed_emax, tau, 'solve')
    write_resfort_initialization(*args)

# Compile
os.system('python driver.py')

# Test the functionality of the executable for varying number of slaves and
# varying number of model specifications.
import numpy as np
# TODO: Only checks if executing without problem, later test if same result.
for i in range(100000):
    constr = dict()
    constr['periods'] = np.random.randint(2, 5)
    generate_init(constr)
    base = None
    for num_slaves in [1, 2, 3]:

        cmd = 'mpiexec ./master ' + str(num_slaves)
        assert os.system(cmd) == 0

        if base is None:
            base = np.loadtxt('.eval.resfort.dat')

        assert base == np.loadtxt('.eval.resfort.dat')