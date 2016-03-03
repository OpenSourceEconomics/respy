#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
import statsmodels.api as sm
import numpy as np

import numpy as np
import random
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
import pandas as pd
from robupy.python.py.auxiliary import get_total_value
from robupy.auxiliary import replace_missing_values
from robupy.python.py.risk import get_payoffs_risk
from robupy.auxiliary import create_disturbances

from modules.auxiliary import compile_package
from robupy.constants import MISSING_FLOAT, TINY_FLOAT
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import generate_init

from robupy.tests.random_init import print_random_dict
from robupy import *

from robupy.python.py.auxiliary import opt_get_model_parameters
from robupy.python.py.auxiliary import opt_get_optim_parameters

from modules.auxiliary import write_interpolation_grid

# Read in baseline initialization file
#compile_package('--fortran --debug', False)

np.random.seed(123)

# Generate random initialization file
#generate_init()

robupy_obj = read('test.robupy.ini')

solve(robupy_obj)

simulate(robupy_obj)

data_frame = process('data.robupy.dat', robupy_obj)

data_array = data_frame.as_matrix()

is_ambiguous = robupy_obj.get_attr('is_ambiguous')
num_periods = robupy_obj.get_attr('num_periods')
is_debug = robupy_obj.get_attr('is_debug')
num_sims = robupy_obj.get_attr('num_sims')
seed_estimation = robupy_obj.get_attr('seed_estimation')

model_paras = robupy_obj.get_attr('model_paras')

eps_cholesky = model_paras['eps_cholesky']

standard_deviates = create_disturbances(num_sims, seed_estimation,
                                               eps_cholesky,
                                               is_ambiguous,
                                               num_periods,
                                               is_debug, 'estimation')

print(eps_cholesky.shape)
model_paras = robupy_obj.get_attr("model_paras")

coeffs_a = model_paras['coeffs_a']
coeffs_b = model_paras['coeffs_b']

coeffs_edu = model_paras['coeffs_edu']
coeffs_home = model_paras['coeffs_home']
shocks = model_paras['shocks']

print(eps_cholesky.shape, 'here')

x = opt_get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                             shocks, eps_cholesky, is_debug)

print(eps_cholesky.shape)

coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, eps_cholesky = \
    opt_get_model_parameters(x, is_debug)


print(type(coeffs_a))


periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')
mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')
periods_emax = robupy_obj.get_attr('periods_emax')
num_periods = robupy_obj.get_attr('num_periods')
num_agents = robupy_obj.get_attr('num_agents')
states_all = robupy_obj.get_attr('states_all')
edu_start = robupy_obj.get_attr('edu_start')
num_sims = robupy_obj.get_attr('num_sims')
edu_max = robupy_obj.get_attr('edu_max')
delta = robupy_obj.get_attr('delta')

is_interpolated = robupy_obj.get_attr('is_interpolated')
is_python = robupy_obj.get_attr('is_python')
level = robupy_obj.get_attr('level')
measure = robupy_obj.get_attr('measure')
is_interpolated = robupy_obj.get_attr('is_interpolated')

is_ambiguous = robupy_obj.get_attr('is_ambiguous')

num_periods = robupy_obj.get_attr('num_periods')

num_points = robupy_obj.get_attr('num_points')

num_draws = robupy_obj.get_attr('num_draws')

edu_start = robupy_obj.get_attr('edu_start')

init_dict = robupy_obj.get_attr('init_dict')

is_python = robupy_obj.get_attr('is_python')

is_debug = robupy_obj.get_attr('is_debug')

measure = robupy_obj.get_attr('measure')

edu_max = robupy_obj.get_attr('edu_max')

min_idx = robupy_obj.get_attr('min_idx')


store = robupy_obj.get_attr('store')

delta = robupy_obj.get_attr('delta')

level = robupy_obj.get_attr('level')

seed_solution = robupy_obj.get_attr('seed_solution')



# Construct auxiliary objects
args = opt_get_model_parameters(x, is_debug)

robupy_obj.update_model_paras(*args)

base = evaluate(robupy_obj, data_array)


np.testing.assert_almost_equal(base, 0.18991613969942756)


# Start of unit test design.
constraints = dict()
constraints['debug'] = True

init_dict = generate_random_dict(constraints)

# Print to dictionary
print_random_dict(init_dict)

# Perform toolbox actions
robupy_obj = read('test.robupy.ini')

robupy_obj = solve(robupy_obj)

simulate(robupy_obj)