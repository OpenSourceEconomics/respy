""" This module contains the interface for the evaluation of the criterion
function.
"""

# project library
from robupy.evaluate.evaluate_auxiliary import check_evaluation
from robupy.evaluate.evaluate_python import pyth_evaluate

from robupy.fortran.fortran import fort_evaluate

from robupy.shared.auxiliary import distribute_class_attributes
from robupy.shared.auxiliary import distribute_model_paras
from robupy.shared.auxiliary import create_draws
from robupy.shared.auxiliary import add_solution

from robupy.shared.auxiliary import replace_missing_values

''' Main function
'''


def evaluate(robupy_obj, data_frame):
    """ Evaluate likelihood function.
    """
    # Antibugging
    assert robupy_obj.get_attr('is_solved')
    assert robupy_obj.get_status()

    # Distribute class attributes
    periods_payoffs_systematic, mapping_state_idx, periods_emax, model_paras, \
        num_periods, num_agents, states_all, edu_start, is_python, seed_data, \
        is_debug, file_sim, edu_max, delta, is_deterministic, version, \
        num_draws_prob, seed_prob, num_draws_emax, seed_emax, is_interpolated, \
        is_ambiguous, num_points, is_myopic, min_idx, measure, level, store = \
            distribute_class_attributes(robupy_obj,
                'periods_payoffs_systematic', 'mapping_state_idx',
                'periods_emax', 'model_paras', 'num_periods', 'num_agents',
                'states_all', 'edu_start', 'is_python', 'seed_data',
                'is_debug', 'file_sim', 'edu_max', 'delta',
                'is_deterministic', 'version', 'num_draws_prob', 'seed_prob',
                'num_draws_emax', 'seed_emax', 'is_interpolated',
                'is_ambiguous', 'num_points', 'is_myopic', 'min_idx', 'measure',
                'level', 'store')

    # Check the dataset against the initialization files
    assert check_evaluation('in', data_frame, robupy_obj, is_deterministic)



    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        distribute_model_paras(model_paras, is_debug)

    # Draw standard normal deviates for choice probability integration
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug, 'prob', shocks_cholesky)

    # Draw standard normal deviates for EMAX integration
    data_array = data_frame.as_matrix()

    args = (periods_payoffs_systematic, mapping_state_idx, periods_emax,
        states_all, shocks_cov, shocks_cholesky, is_deterministic,
        num_periods, edu_start, edu_max, delta, data_array, num_agents,
        num_draws_prob, periods_draws_prob)

    # Select appropriate interface
    if version == 'FORTRAN':

        crit_val = fort_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, seed_emax, is_debug,
        min_idx, measure, edu_max, delta, level, num_draws_prob, num_agents,
        seed_prob, seed_data, 'evaluate', data_frame)

    elif version == 'PYTHON':

        crit_val = pyth_evaluate(*args)

    elif version == 'F2PY':

        from robupy.fortran.f2py_library import f2py_evaluate
        crit_val = f2py_evaluate(*args)

    else:

        raise NotImplementedError


    # Checks
    assert check_evaluation('out', crit_val)

    # Finishing
    return crit_val
