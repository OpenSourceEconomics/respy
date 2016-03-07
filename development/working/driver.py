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
from robupy.auxiliary import create_disturbances
from robupy.auxiliary import distribute_model_paras

from robupy.python.solve_python import solve_python_bare
from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

# testing battery
from modules.auxiliary import compile_package



compile_package('--fortran --debug', False)
import robupy.python.f2py.f2py_debug as fort
import random
for _ in range(1000):

    print(_)
    seed = random.randint(a=0, b=100000)

    np.random.seed(seed)

    generate_init()

    # Perform toolbox actions
    robupy_obj = read('test.robupy.ini')

    robupy_obj = solve(robupy_obj)

    # Extract class attributes
    states_number_period = robupy_obj.get_attr('states_number_period')

    is_interpolated = robupy_obj.get_attr('is_interpolated')

    seed_solution = robupy_obj.get_attr('seed_solution')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    model_paras = robupy_obj.get_attr('model_paras')

    num_periods = robupy_obj.get_attr('num_periods')

    num_points = robupy_obj.get_attr('num_points')

    edu_start = robupy_obj.get_attr('edu_start')

    is_python = robupy_obj.get_attr('is_python')

    num_draws = robupy_obj.get_attr('num_draws')

    is_debug = robupy_obj.get_attr('is_debug')

    measure = robupy_obj.get_attr('measure')

    edu_max = robupy_obj.get_attr('edu_max')

    min_idx = robupy_obj.get_attr('min_idx')

    delta = robupy_obj.get_attr('delta')

    level = robupy_obj.get_attr('level')

    # Extract auxiliary objects
    max_states_period = max(states_number_period)

    eps_cholesky = model_paras['eps_cholesky']

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, eps_cholesky = \
         distribute_model_paras(model_paras, is_debug)

    # Get set of disturbances
    periods_eps_relevant = create_disturbances(num_draws, seed_solution,
        eps_cholesky, is_ambiguous, num_periods, is_debug, 'solution')

    # Baseline input arguments.
    base_args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
            edu_max, delta, edu_start, is_debug, is_interpolated,
            level, measure, min_idx, num_draws, num_periods, num_points,
            is_ambiguous, periods_eps_relevant]

    base = None
    for version in ['PYTHON', 'F2PY', 'FORTRAN']:
        if version in ['F2PY', 'PYTHON']:
            # Modifications to input arguments
            args = base_args + [is_python]
            # Get PYTHON/F2PY results
            ret_args = solve_python_bare(*args)
        else:
            # Modifications to input arguments
            args = base_args + [max_states_period]
            # Get FORTRAN results
            ret_args = fort.wrapper_solve_fortran_bare(*args)
        # Collect baseline information
        if base is None:
            base = ret_args
        # Check results
        for i in range(7):
            np.testing.assert_equal(base[i], replace_missing_values(ret_args[i]))


