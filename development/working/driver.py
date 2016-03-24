#!/usr/bin/env python
""" I will now try to run some estimations.
"""


# standard library
import numpy as np
from scipy.optimize import minimize
import sys
import os

# ROOT DIRECTORY
sys.path.insert(0, os.environ['ROBUPY'])
# ROOT DIRECTORY
ROOT_DIR = os.environ['ROBUPY']
ROOT_DIR = ROOT_DIR + '/robupy/tests'

sys.path.insert(0, ROOT_DIR)

# testing codes
from codes.auxiliary import cleanup_robupy_package
from codes.auxiliary import build_testing_library
from codes.auxiliary import build_robupy_package
from codes.auxiliary import distribute_model_description
from robupy.estimate.estimate_auxiliary import opt_get_optim_parameters
from robupy.estimate.estimate_auxiliary import logging_optimization
from robupy.estimate.estimate_python import pyth_criterion
from robupy.shared.auxiliary import create_draws

from robupy.shared.auxiliary import distribute_class_attributes
from robupy.shared.auxiliary import distribute_model_paras

from robupy import simulate, read, solve, process, evaluate, estimate
build_robupy_package(False)

robupy_obj = read('test.robupy.ini')

# First, I simulate a dataset.
robupy_obj = solve(robupy_obj)

val = robupy_obj.get_attr('periods_emax')[0, 0]
#np.testing.assert_allclose(3.664605209230335, val)

simulate(robupy_obj)
val = evaluate(robupy_obj, process(robupy_obj))
np.testing.assert_allclose(2.992618550039753, val)

data_frame = process(robupy_obj)

estimate(robupy_obj, data_frame)

# Distribute class attributes
periods_payoffs_systematic, mapping_state_idx, periods_emax, model_paras, \
    num_periods, num_agents, states_all, edu_start, is_python, seed_data, \
    is_debug, file_sim, edu_max, delta, num_draws_prob, seed_prob, \
    num_draws_emax, seed_emax, level, measure, min_idx, is_ambiguous, \
    is_deterministic, is_myopic, is_interpolated, num_points, version = \
    distribute_class_attributes(robupy_obj,
        'periods_payoffs_systematic', 'mapping_state_idx',
        'periods_emax', 'model_paras', 'num_periods', 'num_agents',
        'states_all', 'edu_start', 'is_python', 'seed_data',
        'is_debug', 'file_sim', 'edu_max', 'delta', 'num_draws_prob',
        'seed_prob', 'num_draws_emax', 'seed_emax', 'level', 'measure',
        'min_idx', 'is_ambiguous', 'is_deterministic', 'is_myopic',
        'is_interpolated', 'num_points', 'version')

# Auxiliary objects
shocks_cholesky = model_paras['shocks_cholesky']

# Draw standard normal deviates for the solution and evaluation step.
periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
    is_debug, 'prob', shocks_cholesky)

periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
    is_debug, 'emax', shocks_cholesky)

# Construct starting values
coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
    distribute_model_paras(model_paras, is_debug)

x0 = opt_get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
    shocks_cov, shocks_cholesky, is_debug)

data_array = data_frame.as_matrix()

from robupy.fortran.f2py_library import f2py_criterion
from robupy.estimate.estimate_python import pyth_criterion

args =(x0, data_array, edu_max, delta, edu_start, is_debug, \
    is_interpolated, level, measure, min_idx, num_draws_emax, num_periods, \
    num_points, is_ambiguous, periods_draws_emax, is_deterministic, \
    is_myopic, num_agents, num_draws_prob, periods_draws_prob)

pyth = pyth_criterion(*args)

f2py = f2py_criterion(*args)

print(f2py, pyth)