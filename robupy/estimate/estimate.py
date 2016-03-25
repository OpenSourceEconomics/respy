""" This module contains the interface for the estimation of the criterion
function.
"""

# standard library
from scipy.optimize import minimize


# project library
from robupy.estimate.estimate_auxiliary import opt_get_optim_parameters
from robupy.estimate.estimate_auxiliary import logging_optimization
from robupy.estimate.estimate_python import pyth_criterion

from robupy.shared.auxiliary import distribute_class_attributes
from robupy.shared.auxiliary import distribute_model_paras
from robupy.shared.auxiliary import create_draws

from robupy.fortran.f2py_library import f2py_criterion

''' Main function
'''

def all_done():
    print('all_done()')


def estimate(robupy_obj, data_frame):
    """ Estimate the model
    """
    # Distribute class attributes
    periods_payoffs_systematic, mapping_state_idx, periods_emax, model_paras, \
        num_periods, num_agents, states_all, edu_start, seed_data, \
        is_debug, file_sim, edu_max, delta, num_draws_prob, seed_prob, \
        num_draws_emax, seed_emax, level, measure, min_idx, is_ambiguous, \
        is_deterministic, is_myopic, is_interpolated, num_points, version = \
            distribute_class_attributes(robupy_obj,
                'periods_payoffs_systematic', 'mapping_state_idx',
                'periods_emax', 'model_paras', 'num_periods', 'num_agents',
                'states_all', 'edu_start', 'seed_data', 'is_debug',
                'file_sim', 'edu_max', 'delta', 'num_draws_prob',
                'seed_prob', 'num_draws_emax', 'seed_emax', 'level', 'measure',
                'min_idx', 'is_ambiguous', 'is_deterministic', 'is_myopic',
                'is_interpolated', 'num_points', 'version')

    # Auxiliary objects
    shocks_cholesky = model_paras['shocks_cholesky']

    # Draw standard normal deviates for the solution and evaluation step.
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug, 'prob', shocks_cholesky)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug, 'emax', shocks_cholesky)

    # Construct starting values
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        distribute_model_paras(model_paras, is_debug)

    x0 = opt_get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, shocks_cholesky, is_debug)



    # Start logging
    x0[0] = 0.25

    data_array = data_frame.as_matrix()

    # Collect arguments for optimization
    args = (data_array, edu_max, delta, edu_start, is_debug,
        is_interpolated, level, measure, min_idx, num_draws_emax, num_periods,
        num_points, is_ambiguous, periods_draws_emax, is_deterministic,
        is_myopic, num_agents, num_draws_prob, periods_draws_prob)

    if version == 'PYTHON':
        crit_val = pyth_criterion(x0, *args)
    elif version in ['F2PY', 'FORTRAN']:
        crit_val = f2py_criterion(x0, *args)
    else:
        raise NotImplementedError

    logging_optimization('start', crit_val, x0)

    optimizer_args = dict()
    optimizer_args['maxiter'] = 0
    optimizer_args['maxfun'] = 1

    minimize(pyth_criterion, x0, method='BFGS', args=args,
        options=optimizer_args)



