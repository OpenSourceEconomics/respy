""" This module provides the interface to the functionality needed to solve the
model with FORTRAN.
"""
# standard library
import os

# module-wide variables
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

# project library
from robupy.fortran.auxiliary import _write_robufort_initialization
from robupy.fortran.auxiliary import _add_results

''' Main function
'''


def solve_fortran(robupy_obj):
    """ Solve dynamic programming using FORTRAN.
    """

   # Distribute class attributes
    model_paras = robupy_obj.get_attr('model_paras')

    coeffs_a = model_paras['coeffs_a']
    coeffs_b = model_paras['coeffs_b']
    coeffs_home = model_paras['coeffs_home']
    coeffs_edu = model_paras['coeffs_edu']
    shocks_cov = model_paras['shocks_cov']

    # Auxiliary objects
    is_deterministic = robupy_obj.get_attr('is_deterministic')

    is_interpolated = robupy_obj.get_attr('is_interpolated')

    num_draws_prob = robupy_obj.get_attr('num_draws_prob')

    num_draws_emax = robupy_obj.get_attr('num_draws_emax')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    num_periods = robupy_obj.get_attr('num_periods')

    num_points = robupy_obj.get_attr('num_points')

    num_agents = robupy_obj.get_attr('num_agents')

    seed_prob = robupy_obj.get_attr('seed_prob')

    seed_data = robupy_obj.get_attr('seed_data')

    is_myopic = robupy_obj.get_attr('is_myopic')

    edu_start = robupy_obj.get_attr('edu_start')

    seed_emax = robupy_obj.get_attr('seed_emax')

    is_debug = robupy_obj.get_attr('is_debug')

    min_idx = robupy_obj.get_attr('min_idx')

    measure = robupy_obj.get_attr('measure')

    edu_max = robupy_obj.get_attr('edu_max')

    delta = robupy_obj.get_attr('delta')

    level = robupy_obj.get_attr('level')

    # Prepare ROBUFORT execution
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_prob, num_draws_emax,
        is_ambiguous, num_periods, num_points, num_agents, seed_prob,
        seed_data, is_myopic, edu_start, seed_emax, is_debug, min_idx,
        measure, edu_max, delta, level, 'solve')

    _write_robufort_initialization(*args)

    # Call executable
    os.system('"' + PACKAGE_PATH + '/bin/robufort"')

    # Add results
    robupy_obj, _ = _add_results(robupy_obj, 'solve')

    # Finishing
    return robupy_obj

