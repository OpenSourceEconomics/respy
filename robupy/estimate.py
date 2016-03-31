""" This module contains the interface for the estimation of the criterion
function.
"""

# standard library


# project library
from robupy.python.estimate.estimate_auxiliary import get_optim_parameters
from robupy.python.estimate.estimate_auxiliary import check_input

from robupy.python.shared.shared_auxiliary import distribute_class_attributes
from robupy.python.shared.shared_auxiliary import distribute_model_paras
from robupy.python.shared.shared_auxiliary import create_draws

from robupy.python.estimate.estimate_wrapper import OptimizationClass

''' Main function
'''

def estimate(robupy_obj, data_frame):
    """ Estimate the model
    """
    # Antibugging
    assert check_input(robupy_obj, data_frame)

    # Distribute class attributes
    model_paras, num_periods, num_agents, edu_start, seed_data, \
        is_debug, file_sim, edu_max, delta, num_draws_prob, seed_prob, \
        num_draws_emax, seed_emax, level, measure, min_idx, is_ambiguous, \
        is_deterministic, is_myopic, is_interpolated, num_points, version, \
        maxiter, optimizer = \
            distribute_class_attributes(robupy_obj,
                'model_paras', 'num_periods', 'num_agents', 'edu_start',
                'seed_data', 'is_debug', 'file_sim', 'edu_max', 'delta',
                'num_draws_prob', 'seed_prob', 'num_draws_emax', 'seed_emax',
                'level', 'measure', 'min_idx', 'is_ambiguous',
                'is_deterministic', 'is_myopic', 'is_interpolated',
                'num_points', 'version', 'maxiter', 'optimizer')

    # Auxiliary objects
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        distribute_model_paras(model_paras, is_debug)

    # Draw standard normal deviates for the solution and evaluation step.
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug, 'prob', shocks_cholesky)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug, 'emax', shocks_cholesky)

    # Construct starting values
    x0 = get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, shocks_cholesky, is_debug)

    data_array = data_frame.as_matrix()

    # Collect arguments that are required for the criterion function. These
    # must be in the correct order already.
    args = (is_deterministic, is_interpolated, num_draws_emax,is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug, measure,
        edu_max, min_idx, delta, level, data_array, num_agents,
        num_draws_prob, periods_draws_emax, periods_draws_prob)

    # Setup optimization class, which handles all the details depending on the
    # request.
    opt_obj = OptimizationClass()

    opt_obj.set_attr('args', args)

    opt_obj.set_attr('optimizer', optimizer)

    opt_obj.set_attr('version', version)

    opt_obj.set_attr('maxiter', maxiter)

    opt_obj.lock()

    # Perform optimization.
    x, fval = opt_obj.optimize(x0)

    print(x)







