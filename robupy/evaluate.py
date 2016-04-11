""" This module contains the interface for the evaluation of the criterion
function.
"""

# project library
from robupy.python.evaluate.evaluate_python import pyth_evaluate
from robupy.fortran.f2py_library import f2py_evaluate

from robupy.fortran.fortran import fort_evaluate

from robupy.python.evaluate.evaluate_auxiliary import check_input
from robupy.python.evaluate.evaluate_auxiliary import check_output

from robupy.python.shared.shared_auxiliary import dist_class_attributes
from robupy.python.shared.shared_auxiliary import dist_model_paras
from robupy.python.shared.shared_auxiliary import create_draws
from robupy.python.shared.shared_auxiliary import cut_dataset

''' Main function
'''


def evaluate(robupy_obj, data_frame):
    """ Evaluate the criterion function.
    """

    # If required, cut the dataset to only contain the number of agents
    # used in the estimation.
    data_frame = cut_dataset(robupy_obj, data_frame)

    # Antibugging
    assert check_input(robupy_obj, data_frame)

    # Distribute class attributes
    model_paras, num_periods, num_agents_est, edu_start, seed_sim, is_debug, \
        edu_max, delta, is_deterministic, version, num_draws_prob, seed_prob, \
        num_draws_emax, seed_emax, is_interpolated, is_ambiguous, num_points, \
        is_myopic, min_idx, level, tau = \
            dist_class_attributes(robupy_obj,
                'model_paras', 'num_periods', 'num_agents_est', 'edu_start',
                'seed_sim', 'is_debug', 'edu_max', 'delta', 'is_deterministic',
                'version', 'num_draws_prob', 'seed_prob', 'num_draws_emax',
                'seed_emax', 'is_interpolated', 'is_ambiguous', 'num_points',
                'is_myopic', 'min_idx', 'level', 'tau')

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, \
        shocks_cholesky = dist_model_paras(model_paras, is_debug)

    # Draw standard normal deviates for choice probabilities and expected
    # future values.
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug)

    # Transform dataset for interface
    data_array = data_frame.as_matrix()

    base_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug,
        edu_max, min_idx, delta, level, data_array, num_agents_est,
        num_draws_prob, tau)

    # Select appropriate interface
    if version == 'FORTRAN':
        args = base_args + (seed_emax, seed_prob)
        crit_val = fort_evaluate(*args)
    elif version == 'PYTHON':
        args = base_args + (periods_draws_emax, periods_draws_prob)
        crit_val = pyth_evaluate(*args)
    elif version == 'F2PY':
        args = base_args + (periods_draws_emax, periods_draws_prob)
        crit_val = f2py_evaluate(*args)
    else:
        raise NotImplementedError

    # Checks
    assert check_output(crit_val)

    # Finishing
    return crit_val
