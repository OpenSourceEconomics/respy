from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.python.evaluate.evaluate_python import pyth_evaluate


def pyth_criterion(x, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
        data_array, num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob):
    """ This function provides the wrapper for optimization routines.
    """
    args = (is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_start, is_debug, edu_max, min_idx, delta, data_array,
        num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob)

    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky \
        = dist_optim_paras(x, is_debug)

    crit_val = pyth_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, *args)

    return crit_val
