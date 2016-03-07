#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
import numpy as np

import scipy
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# RobuPy library
from robupy import *
from robupy.auxiliary import replace_missing_values

from robupy.auxiliary import opt_get_model_parameters
from robupy.auxiliary import opt_get_optim_parameters
from robupy.auxiliary import distribute_model_paras

from robupy.python.solve_python import solve_python_bare

# testing battery
from modules.auxiliary import write_disturbances
from modules.auxiliary import compile_package

# Read in baseline initialization file
compile_package('--fortran --debug', False)


np.random.seed(123)

import robupy.python.f2py.f2py_debug as fort

# Draw random requests for testing purposes.
num_draws = np.random.random_integers(2, 1000)
dim = np.random.random_integers(1, 6)
mean = np.random.uniform(-0.5, 0.5, (dim))

matrix = (np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim))
cov = np.dot(matrix, matrix.T)

# Singular Value Decomposition
py = scipy.linalg.svd(matrix)[0]
f90 = fort.wrapper_svd(matrix, dim)[0]

for i in range(3):
    np.testing.assert_allclose(py[i], f90[i], rtol=1e-05, atol=1e-06)


''' F
'''

robupy_obj = read('test.robupy.ini')

init_dict = robupy_obj.get_attr('init_dict')
# Write out disturbances to align the three implementations.
write_disturbances(init_dict)

solve(robupy_obj)

# The idea is to now set up a unit test for solve_python_bare ....
# TODO: I really want exactly the same input arguments for the unit testing
# between FORTRAN and PYTHON

is_interpolated = robupy_obj.get_attr('is_interpolated')

seed_solution = robupy_obj.get_attr('seed_solution')

is_ambiguous = robupy_obj.get_attr('is_ambiguous')

num_periods = robupy_obj.get_attr('num_periods')

model_paras = robupy_obj.get_attr('model_paras')

num_points = robupy_obj.get_attr('num_points')

num_draws = robupy_obj.get_attr('num_draws')

edu_start = robupy_obj.get_attr('edu_start')

is_python = robupy_obj.get_attr('is_python')

is_debug = robupy_obj.get_attr('is_debug')

measure = robupy_obj.get_attr('measure')

edu_max = robupy_obj.get_attr('edu_max')

min_idx = robupy_obj.get_attr('min_idx')

store = robupy_obj.get_attr('store')

delta = robupy_obj.get_attr('delta')

level = robupy_obj.get_attr('level')

states_number_period = robupy_obj.get_attr('states_number_period')

max_states_period = max(states_number_period)



# Distribute model parameters
coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, eps_cholesky = \
    distribute_model_paras(model_paras, is_debug)

mapping_state_idx, periods_emax, periods_future_payoffs, \
periods_payoffs_ex_post, periods_payoffs_systematic, states_all, \
states_number_period = solve_python_bare(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
        eps_cholesky, edu_max, delta, edu_start, is_debug, is_interpolated,
        level, measure, min_idx, num_draws, num_periods, num_points,
        is_ambiguous, seed_solution, is_python)


fort_emax = robupy_obj.get_attr('periods_emax')


np.testing.assert_almost_equal(fort_emax, periods_emax)

print(periods_emax)
print(fort_emax)

args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
    eps_cholesky, edu_max, delta, edu_start, is_debug, is_interpolated,
    level, measure, min_idx, num_draws, num_periods, num_points,
    is_ambiguous, seed_solution]

# Break in design, maybe remove later ...
is_zero = True
args += [is_zero, max_states_period]

f90 = fort.wrapper_solve_fortran_bare(*args)

py = solve_python_bare(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
        eps_cholesky, edu_max, delta, edu_start, is_debug, is_interpolated,
        level, measure, min_idx, num_draws, num_periods, num_points,
        is_ambiguous, seed_solution, is_python)


print()
print(py[0])
print()
print(f90[0])

#for i in range(7):
#    print(i)
# TODO: Randomness still a problem, try further to fully align interface.
np.testing.assert_equal(py[4], replace_missing_values(f90[4]))
# Payoff ex post
np.testing.assert_equal(py[3], replace_missing_values(f90[3]))

