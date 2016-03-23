""" This module contains the interface for the evaluation of the criterion
function.
"""

# standard library
import numpy as np

# project library
from robupy.fortran.evaluate_fortran import evaluate_fortran
from robupy.evaluate.evaluate_python import evaluate_python

from robupy.shared.auxiliary import distribute_class_attributes
from robupy.shared.auxiliary import distribute_model_paras
from robupy.shared.auxiliary import check_dataset
from robupy.shared.auxiliary import create_draws

''' Main function
'''


def evaluate(robupy_obj, data_frame):
    """ Evaluate likelihood function.
    """
    # Distribute class attributes
    periods_payoffs_systematic, mapping_state_idx, periods_emax, model_paras, \
        num_periods, num_agents, states_all, edu_start, is_python, seed_data, \
        is_debug, file_sim, edu_max, delta, is_deterministic, version, \
        num_draws_prob, seed_prob, num_draws_emax, seed_emax, is_interpolated, \
        is_ambiguous, num_points, is_myopic, min_idx, measure, level = \
            distribute_class_attributes(robupy_obj,
                'periods_payoffs_systematic', 'mapping_state_idx',
                'periods_emax', 'model_paras', 'num_periods', 'num_agents',
                'states_all', 'edu_start', 'is_python', 'seed_data',
                'is_debug', 'file_sim', 'edu_max', 'delta',
                'is_deterministic', 'version', 'num_draws_prob', 'seed_prob',
                'num_draws_emax', 'seed_emax', 'is_interpolated',
                'is_ambiguous', 'num_points', 'is_myopic', 'min_idx', 'measure',
                'level')

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
        likl = evaluate_fortran(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cov, is_deterministic, is_interpolated, num_draws_emax,
            is_ambiguous, num_periods, num_points, is_myopic, edu_start,
            seed_emax, is_debug, min_idx, measure, edu_max, delta, level,
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
