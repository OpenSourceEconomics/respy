""" This module contains the python implementation of the criterion function.
"""

# standard library
from respy.python.estimate.estimate_auxiliary import dist_optim_paras
from respy.python.evaluate.evaluate_python import pyth_evaluate


def pyth_criterion(x, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
        data_array, num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob):
    """ This function provides the wrapper for optimization routines.
    """
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
