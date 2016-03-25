from robupy.estimate.estimate_auxiliary import opt_get_model_parameters

from robupy.solve.solve_python import pyth_solve
from robupy.evaluate.evaluate_python import pyth_evaluate

def pyth_criterion(x, is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug, measure,
        edu_max, min_idx, delta, level, data_array, num_agents, num_draws_prob,
        periods_draws_emax, periods_draws_prob):
    """ This function provides the wrapper for optimization routines.
    """
    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        opt_get_model_parameters(x, is_debug)


    crit_val = pyth_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug, measure,
        edu_max, min_idx, delta, level, data_array, num_agents, num_draws_prob,
        shocks_cholesky, periods_draws_emax, periods_draws_prob)

    # Finishing
    return crit_val