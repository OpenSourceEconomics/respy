""" This module contains the interface for the estimation of the criterion
function.
"""

# standard library


# project library
from robupy.python.estimate.estimate_auxiliary import get_optim_paras
from robupy.python.estimate.estimate_auxiliary import check_input

from robupy.python.shared.shared_auxiliary import dist_class_attributes
from robupy.python.shared.shared_auxiliary import dist_model_paras
from robupy.python.shared.shared_auxiliary import create_draws
from robupy.python.shared.shared_auxiliary import cut_dataset

from robupy.python.estimate.estimate_wrapper import OptimizationClass

''' Main function
'''


def estimate(robupy_obj, data_frame):
    """ Estimate the model
    """

    # Cut dataset to size in case more agents are passed in than are actually
    # used in the estimation.
    data_frame = cut_dataset(robupy_obj, data_frame)

    # Antibugging
    assert check_input(robupy_obj, data_frame)

    # Distribute class attributes
    model_paras, num_periods, num_agents_est, edu_start, seed_sim, \
        is_debug, file_sim, edu_max, delta, num_draws_prob, seed_prob, \
        num_draws_emax, seed_emax, level, min_idx, is_ambiguous, \
        is_deterministic, is_myopic, is_interpolated, num_points, version, \
        maxiter, optimizer, tau, paras_fixed, file_opt = \
            dist_class_attributes(robupy_obj,
                'model_paras', 'num_periods', 'num_agents_est', 'edu_start',
                'seed_sim', 'is_debug', 'file_sim', 'edu_max', 'delta',
                'num_draws_prob', 'seed_prob', 'num_draws_emax', 'seed_emax',
                'level', 'min_idx', 'is_ambiguous',
                'is_deterministic', 'is_myopic', 'is_interpolated',
                'num_points', 'version', 'maxiter', 'optimizer', 'tau',
                'paras_fixed', 'file_opt')

    # Auxiliary objects
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    # Draw standard normal deviates for the solution and evaluation step.
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug)

    # Construct starting values
    x_free_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, 'free', paras_fixed, is_debug)

    x_all_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, 'all', paras_fixed, is_debug)

    data_array = data_frame.as_matrix()

    # Collect arguments that are required for the criterion function. These
    # must be in the correct order already.
    args = (is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug,
        edu_max, min_idx, delta, level, data_array, num_agents_est,
        num_draws_prob, tau, periods_draws_emax, periods_draws_prob)

    # Setup optimization class, which handles all the details depending on the
    # request.
    opt_obj = OptimizationClass()

    opt_obj.set_attr('args', args)

    opt_obj.set_attr('x_info', (x_all_start, paras_fixed))

    opt_obj.set_attr('optimizer', optimizer)

    opt_obj.set_attr('file_opt', file_opt)

    opt_obj.set_attr('version', version)

    opt_obj.set_attr('maxiter', maxiter)

    opt_obj.lock()

    # Perform optimization.
    x, val = opt_obj.optimize(x_free_start)

    # Finishing
    return x, val







