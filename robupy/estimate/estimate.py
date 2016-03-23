""" This module contains the interface for the estimation of the criterion
function.
"""

# standard library
from scipy.optimize import minimize

# project library
from robupy.estimate.estimate_auxiliary import opt_get_optim_parameters

from robupy.shared.auxiliary import distribute_model_paras
from robupy.shared.auxiliary import create_draws



# TODO: Some nice debugging is in order.as

def criterion(x, data_frame, edu_max, delta, edu_start, is_debug,
        is_interpolated, level, measure, min_idx, num_draws_emax,
        num_periods, num_points, is_ambiguous, periods_draws_emax,
        is_deterministic, is_myopic, num_agents, num_draws_prob, data_array,
            periods_draws_prob, is_python):
    #
    # coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
    #     opt_get_model_parameters(x, is_debug)
    #
    # # Solve model for given parametrization
    # args = solve_python(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
    #     shocks_cholesky, is_deterministic, is_interpolated, num_draws_emax,
    #     periods_draws_emax, is_ambiguous, num_periods, num_points, edu_start,
    #     is_myopic, is_debug, measure, edu_max, min_idx, delta, level,
    #     is_python)
    #
    # # Distribute return arguments from solution run
    # mapping_state_idx, periods_emax, periods_payoffs_future = args[:3]
    # periods_payoffs_ex_post, periods_payoffs_systematic, states_all = args[3:6]
    #
    # # Evaluate the criterion function
    # likl = evaluate_python(mapping_state_idx, periods_emax,
    #     periods_payoffs_systematic, states_all, shocks_cov, edu_max,
    #     delta, edu_start, num_periods, shocks_cholesky, num_agents,
    #     num_draws_prob, data_array, periods_draws_prob, is_deterministic,
    #     is_python)
    likl = 0.0
    print(likl)
    # Finishing
    return likl

''' Main function
'''

def estimate(robupy_obj, data_frame):
    """ Estimate the model
    """

    data_array = data_frame.as_matrix()

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')
    num_draws_prob = robupy_obj.get_attr('num_draws_prob')
    seed_prob = robupy_obj.get_attr('seed_prob')
    is_debug = robupy_obj.get_attr('is_debug')
    num_draws_emax = robupy_obj.get_attr('num_draws_emax')
    seed_emax = robupy_obj.get_attr('seed_emax')

    model_paras = robupy_obj.get_attr('model_paras')

    shocks_cholesky = model_paras['shocks_cholesky']

    # Draw standard normal deviates for
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug, 'prob', shocks_cholesky)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug, 'emax', shocks_cholesky)

    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
            distribute_model_paras(model_paras, is_debug)

    x0 = opt_get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                                  shocks_cov, shocks_cholesky, is_debug)

    edu_max = robupy_obj.get_attr('edu_max')
    delta = robupy_obj.get_attr('delta')
    edu_start = robupy_obj.get_attr('edu_start')
    is_interpolated = robupy_obj.get_attr('is_interpolated')
    level = robupy_obj.get_attr('level')
    measure = robupy_obj.get_attr('measure')
    min_idx = robupy_obj.get_attr('min_idx')
    num_points = robupy_obj.get_attr('num_points')
    is_ambiguous = robupy_obj.get_attr('is_ambiguous')
    is_deterministic = robupy_obj.get_attr('is_deterministic')
    is_myopic = robupy_obj.get_attr('is_myopic')
    is_python = robupy_obj.get_attr('is_python')

    num_agents = robupy_obj.get_attr('num_agents')

    # Collect arguments for optimization
    args = (data_frame, edu_max, delta, edu_start, is_debug, is_interpolated,
        level, measure, min_idx, num_draws_emax, num_periods, num_points,
        is_ambiguous, periods_draws_emax, is_deterministic, is_myopic,
        num_agents, num_draws_prob, data_array, periods_draws_prob,
        is_python)

    minimize(criterion, x0, method='Powell', args=args)

#    print(x0)
#    x0[0] = 0.25






