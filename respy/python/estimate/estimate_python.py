""" This module contains the python implementation of the criterion function.
"""

import numpy as np

# standard library
from respy.python.estimate.estimate_auxiliary import dist_optim_paras
from respy.python.evaluate.evaluate_python import pyth_evaluate

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var('num_steps', -1)
@static_var('fval_step', np.inf)
def pyth_wrapper(x, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
        data_array, num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob):


    fval = pyth_criterion(x, is_interpolated, num_draws_emax, num_periods,
                num_points_interp, is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
                data_array, num_agents_est, num_draws_prob, tau, periods_draws_emax,
                periods_draws_prob)

    is_start = (pyth_wrapper.fval_step == np.inf)
    is_step = (fval < pyth_wrapper.fval_step)

    # Update static variables
    if True:
        pyth_wrapper.num_steps += 1
        pyth_wrapper.fval_step = fval

    # Logging:
    if True:
        info_start = np.concatenate(([0, fval], x))
        fname = 'opt_info_start.respy.log'
        np.savetxt(open(fname, 'wb'), info_start, fmt='%15.8f')

    if True:
        info_step = np.concatenate(([pyth_wrapper.num_steps, fval], x))
        fname = 'opt_info_step.respy.log'
        np.savetxt(open(fname, 'wb'), info_step, fmt='%15.8f')

    info_current = np.concatenate(([pyth_criterion.num_evals, fval], x))
    fname = 'opt_info_current.respy.log'
    np.savetxt(open(fname, 'wb'), info_current, fmt='%15.8f')

    return fval

@static_var('num_evals', 0)
def pyth_criterion(x, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
        data_array, num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob):
    """ This function provides the wrapper for optimization routines.
    """
    pyth_criterion.num_evals += 1

    # Collect arguments
    args = (is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_start, is_debug, edu_max, min_idx, delta, data_array,
        num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob)

    # Distribute optimization parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky \
        = dist_optim_paras(x, is_debug)

    # Evaluate criterion function
    crit_val = pyth_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, *args)

    # Finishing
    return crit_val
