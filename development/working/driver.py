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

from robupy import simulate, read, solve, process
from robupy.auxiliary import opt_get_model_parameters, opt_get_optim_parameters
from robupy.auxiliary import distribute_model_paras
from robupy.auxiliary import create_disturbances
from robupy.python.solve_python import solve_python_bare
from robupy.python.evaluate_python import _evaluate_python_bare
robupy_obj = read('test.robupy.ini')

# First, I simulate a dataset.
solve(robupy_obj)
sys.exit('preparing eps revamp')
#simulate(robupy_obj)

data_frame = process(robupy_obj)

model_paras = robupy_obj.get_attr('model_paras')

num_periods = robupy_obj.get_attr('num_periods')
num_draws_prob = robupy_obj.get_attr('num_draws_prob')
seed_prob = robupy_obj.get_attr('seed_prob')
is_debug = robupy_obj.get_attr('is_debug')
is_ambiguous = robupy_obj.get_attr('is_ambiguous')
seed_emax = robupy_obj.get_attr('seed_emax')
num_draws_emax = robupy_obj.get_attr('num_draws_emax')

edu_max = robupy_obj.get_attr('edu_max')
delta = robupy_obj.get_attr('delta')
edu_start = robupy_obj.get_attr('edu_start')
is_interpolated = robupy_obj.get_attr('is_interpolated')
level = robupy_obj.get_attr('level')
measure = robupy_obj.get_attr('measure')
min_idx = robupy_obj.get_attr('min_idx')
num_points = robupy_obj.get_attr('num_points')
is_deterministic = robupy_obj.get_attr('is_deterministic')
is_myopic = robupy_obj.get_attr('is_myopic')
is_python = robupy_obj.get_attr('is_python')
num_agents = robupy_obj.get_attr('num_agents')

coeffs_a, coeffs_b, coeffs_edu, coeffs_home, \
    shocks, shocks_cholesky = \
        distribute_model_paras(model_paras, True)
x0 = opt_get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks, shocks_cholesky, True)


# Draw standard normal deviates for S-ML approach
disturbances_prob = create_disturbances(num_periods, num_draws_prob,
         seed_prob, is_debug, 'prob', shocks_cholesky, is_ambiguous)


data_array = data_frame.as_matrix()


def criterion(x, data_array, disturbances_prob):

    assert (isinstance(data_array, np.ndarray))

    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, \
        shocks, shocks_cholesky = \
            opt_get_model_parameters(x, True)


    # Get the relevant set of disturbances. These are standard normal draws
    # in the case of an ambiguous world. This function is located outside the
    # actual bare solution algorithm to ease testing across implementations.
    # TODO: THese need to be adjusted in the case of estimation to alway be
    # TODO: standard normal as well and them moved outside

    disturbances_emax = create_disturbances(num_periods, num_draws_emax,
          seed_emax, is_debug, 'emax', shocks_cholesky, is_ambiguous)

    # Solve model for given parametrization
    args = solve_python_bare(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
         shocks, edu_max, delta, edu_start, is_debug, is_interpolated, level,
        measure, min_idx, num_draws_emax, num_periods, num_points, is_ambiguous,
         disturbances_emax, is_deterministic, is_myopic, is_python)

    # Distribute return arguments from solution run
    mapping_state_idx, periods_emax, periods_payoffs_future = args[:3]
    periods_payoffs_ex_post, periods_payoffs_systematic, states_all = args[3:6]

     # Evaluate the criterion function
    likl = _evaluate_python_bare(mapping_state_idx, periods_emax,
                 periods_payoffs_systematic, states_all, shocks, edu_max,
                 delta, edu_start, num_periods, shocks_cholesky, num_agents,
                 num_draws_prob, data_array, disturbances_prob, is_deterministic,
                 is_python)

    print(likl)

    return likl

criterion(x0, data_array, disturbances_prob)

print(x0)
x0[0] = 0.25
minimize(criterion, x0, method='Powell', args =(data_array, disturbances_prob))