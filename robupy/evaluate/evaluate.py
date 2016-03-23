""" This module contains the interface for the evaluation of the criterion
function.
"""

# standard library
import numpy as np

# project library
from robupy.fortran.evaluate_fortran import evaluate_fortran
from robupy.evaluate.evaluate_python import evaluate_python

from robupy.shared.auxiliary import distribute_model_paras
from robupy.shared.auxiliary import check_dataset
from robupy.shared.auxiliary import create_draws

''' Main function
'''


def evaluate(robupy_obj, data_frame):
    """ Evaluate likelihood function.
    """
    # Distribute class attribute
    is_deterministic = robupy_obj.get_attr('is_deterministic')

    version = robupy_obj.get_attr('version')

    # Distribute class attribute
    is_deterministic = robupy_obj.get_attr('is_deterministic')

    is_interpolated = robupy_obj.get_attr('is_interpolated')

    num_draws_emax = robupy_obj.get_attr('num_draws_emax')

    num_draws_prob = robupy_obj.get_attr('num_draws_prob')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    model_paras = robupy_obj.get_attr('model_paras')

    num_periods = robupy_obj.get_attr('num_periods')

    num_points = robupy_obj.get_attr('num_points')

    num_agents = robupy_obj.get_attr('num_agents')

    seed_prob = robupy_obj.get_attr('seed_prob')

    seed_emax = robupy_obj.get_attr('seed_emax')

    is_python = robupy_obj.get_attr('is_python')

    edu_start = robupy_obj.get_attr('edu_start')

    is_myopic = robupy_obj.get_attr('is_myopic')

    is_debug = robupy_obj.get_attr('is_debug')

    edu_max = robupy_obj.get_attr('edu_max')

    measure = robupy_obj.get_attr('measure')

    min_idx = robupy_obj.get_attr('min_idx')

    level = robupy_obj.get_attr('level')

    delta = robupy_obj.get_attr('delta')

    seed_data = robupy_obj.get_attr('seed_data')

    # Check the dataset against the initialization files
    assert _check_evaluation('in', data_frame, robupy_obj, is_deterministic)

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        distribute_model_paras(model_paras, is_debug)

    # Draw standard normal deviates for choice probability integration
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug, 'prob', shocks_cholesky)

    # Draw standard normal deviates for EMAX integration
    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug, 'emax', shocks_cholesky)

    # Select appropriate interface
    if version == 'FORTRAN':
        likl = evaluate_fortran(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, seed_emax,
            is_debug, min_idx, measure, edu_max, delta, level,
            num_draws_prob, num_agents, seed_prob, seed_data, 'evaluate',
                     data_frame)
    else:
        likl = evaluate_python(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cov, shocks_cholesky, is_deterministic, is_interpolated,
            num_draws_emax, periods_draws_emax, is_ambiguous, num_periods,
            num_points, edu_start, is_myopic, is_debug, measure, edu_max,
            min_idx, delta, level, data_frame,  num_agents, num_draws_prob,
            periods_draws_prob, is_python)

    # Checks
    assert _check_evaluation('out', likl)

    # TODO: Should I give the teh robupyCls is_solved atribute, Am I suing it
    #  at all?
    # Finishing
    return likl

''' Auxiliary functions
'''


def _check_evaluation(str_, *args):
    """ Check integrity of criterion function.
    """
    if str_ == 'out':

        # Distribute input parameters
        likl, = args

        # Check quality
        assert isinstance(likl, float)
        assert np.isfinite(likl)

    elif str_ == 'in':

        # Distribute input parameters
        data_frame, robupy_obj, is_deterministic = args

        # Check quality
        check_dataset(data_frame, robupy_obj)

    # Finishing
    return True
