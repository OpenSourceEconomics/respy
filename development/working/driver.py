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
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# RobuPy library
from robupy import *

from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import create_disturbances
from robupy.auxiliary import distribute_model_paras
from modules.auxiliary import write_interpolation_grid

from robupy.python.solve_python import solve_python_bare
from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init
from robupy.python.evaluate_python import _evaluate_python_bare
# testing battery
from modules.auxiliary import compile_package

compile_package('--fortran --debug', False)

import robupy.python.f2py.f2py_debug as fort

np.random.seed(321)

for _ in range(1):
    constraints = dict()
    constraints['debug'] = True


    generate_init(constraints)
    # Perform toolbox actions
    robupy_obj = read('test.robupy.ini')

    robupy_obj = solve(robupy_obj)

    data_frame = simulate(robupy_obj)

    model_paras = robupy_obj.get_attr('model_paras')
    is_debug = robupy_obj.get_attr('is_debug')


    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, eps_cholesky = \
                 distribute_model_paras(model_paras, is_debug)

    # Distribute class attribute
    is_interpolated = robupy_obj.get_attr('is_interpolated')
    seed_estimation = robupy_obj.get_attr('seed_estimation')
    seed_solution = robupy_obj.get_attr('seed_solution')
    is_ambiguous = robupy_obj.get_attr('is_ambiguous')
    model_paras = robupy_obj.get_attr('model_paras')
    num_periods = robupy_obj.get_attr('num_periods')
    num_points = robupy_obj.get_attr('num_points')
    num_agents = robupy_obj.get_attr('num_agents')
    is_python = robupy_obj.get_attr('is_python')
    edu_start = robupy_obj.get_attr('edu_start')
    num_draws = robupy_obj.get_attr('num_draws')
    is_debug = robupy_obj.get_attr('is_debug')
    num_sims = robupy_obj.get_attr('num_sims')
    edu_max = robupy_obj.get_attr('edu_max')
    measure = robupy_obj.get_attr('measure')
    min_idx = robupy_obj.get_attr('min_idx')
    level = robupy_obj.get_attr('level')
    delta = robupy_obj.get_attr('delta')

    states_number_period = robupy_obj.get_attr('states_number_period')
    max_states_period = max(states_number_period)

    if is_interpolated:
        write_interpolation_grid(num_periods, num_points, states_number_period)

    # Transform dataset to array for easy access

    data_array = data_frame.as_matrix()
        # Draw standard normal deviates for S-ML approach
    standard_deviates = create_disturbances(num_sims, seed_estimation,
            eps_cholesky, is_ambiguous, num_periods, is_debug, 'estimation')

    periods_eps_relevant = create_disturbances(num_draws, seed_solution,
            eps_cholesky, is_ambiguous, num_periods, is_debug, 'solution')

    print('PYTHON')
    py = _evaluate_python_bare(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
            edu_max, delta, edu_start, is_debug, is_interpolated, level, measure,
            min_idx, num_draws, num_periods, num_points, is_ambiguous,
            periods_eps_relevant, eps_cholesky, num_agents, num_sims,
            data_array, standard_deviates, is_python)

    print('FORTRAN')
    f90 = fort.wrapper_evaluate_fortran_bare(coeffs_a, coeffs_b, coeffs_edu,
            coeffs_home, shocks, edu_max, delta, edu_start, is_debug,
            is_interpolated, level, measure, min_idx, num_draws, num_periods,
            num_points, is_ambiguous, periods_eps_relevant, eps_cholesky,
            num_agents, num_sims, data_array, standard_deviates)

    np.testing.assert_allclose(py, 6.276720655476828)
    np.testing.assert_allclose(py, f90)

    try:
        os.unlink('interpolation.txt')

    except:
        pass