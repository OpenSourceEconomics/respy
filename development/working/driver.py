#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
from scipy.stats import norm

import numpy as np

import scipy
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests')
sys.path.insert(0, os.environ['ROBUPY'])

# RobuPy library
from robupy import *

from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import distribute_model_paras
from material.auxiliary import write_interpolation_grid
from material.auxiliary import write_disturbances

from robupy.python.solve_python import solve_python_bare
from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

from robupy.python.evaluate_python import _evaluate_python_bare, evaluate_criterion_function
# testing battery
from material.auxiliary import compile_package
from material.auxiliary import cleanup


print('not recompiling')
#compile_package('--fortran --debug', False)

np.random.seed(123)

robupy_obj = read('test.robupy.ini')

num_periods = robupy_obj.get_attr('num_periods')
max_draws = 1000
write_disturbances(num_periods, max_draws)


robupy_obj = solve(robupy_obj)

data_frame = simulate(robupy_obj)

print(evaluate(robupy_obj, data_frame))