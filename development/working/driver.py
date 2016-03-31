#!/usr/bin/env python
""" I will now try to run some estimations.
"""


# standard library
import os
import sys

# ROOT DIRECTORY
sys.path.insert(0, os.environ['ROBUPY'])
# ROOT DIRECTORY
ROOT_DIR = os.environ['ROBUPY']
ROOT_DIR = ROOT_DIR + '/robupy/tests'

sys.path.insert(0, ROOT_DIR)

# testing codes

from robupy import simulate, read, solve
from robupy.python.estimate.estimate_wrapper import OptimizationClass
from robupy.python.estimate.estimate_auxiliary import get_optim_parameters
from robupy.python.estimate.estimate_auxiliary import check_input

from robupy.python.shared.shared_auxiliary import distribute_class_attributes
from robupy.python.shared.shared_auxiliary import distribute_model_paras
from robupy.python.shared.shared_auxiliary import create_draws

from robupy.python.estimate.estimate_wrapper import OptimizationClass

robupy_obj = read('model.robupy.ini')

# First, I simulate a dataset.
print('starting to solve')
robupy_obj = solve(robupy_obj)

#val = robupy_obj.get_attr('periods_emax')[0, 0]
#np.testing.assert_allclose(1.4963828613937988, val)

print('starting to simulate')
data_frame = simulate(robupy_obj)
#print('starting to estimate')
#äestimate(robupy_obj, process(robupy_obj))

#val = evaluate(robupy_obj, process(robupy_obj))
#np.testing.assert_allclose(2.992618550039753, val)
data_array = data_frame.as_matrix()

model_paras, num_periods, num_agents, edu_start, seed_data, \
    is_debug, file_sim, edu_max, delta, num_draws_prob, seed_prob, \
    num_draws_emax, seed_emax, level, measure, min_idx, is_ambiguous, \
    is_deterministic, is_myopic, is_interpolated, num_points, version, \
    maxiter, optimizer = \
    distribute_class_attributes(robupy_obj,
        'model_paras', 'num_periods', 'num_agents', 'edu_start',
        'seed_data', 'is_debug', 'file_sim', 'edu_max', 'delta',
        'num_draws_prob', 'seed_prob', 'num_draws_emax', 'seed_emax',
        'level', 'measure', 'min_idx', 'is_ambiguous',
        'is_deterministic', 'is_myopic', 'is_interpolated',
        'num_points', 'version', 'maxiter', 'optimizer')

# Auxiliary objects
coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
    distribute_model_paras(model_paras, is_debug)

# Draw standard normal deviates for the solution and evaluation step.
periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
    is_debug, 'prob', shocks_cholesky)

periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
    is_debug, 'emax', shocks_cholesky)

args = (is_deterministic, is_interpolated, num_draws_emax,is_ambiguous,
    num_periods, num_points, is_myopic, edu_start, is_debug, measure,
    edu_max, min_idx, delta, level, data_array, num_agents,
    num_draws_prob, periods_draws_emax, periods_draws_prob)

opt_obj = OptimizationClass()

opt_obj.set_attr('args', args)

opt_obj.set_attr('optimizer', 'SCIPY-BFGS')

opt_obj.set_attr('version', 'FORTRAN')

opt_obj.set_attr('maxiter', 0)

opt_obj.lock()


opt_obj.get_gradient(x)