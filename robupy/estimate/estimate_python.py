from robupy.estimate.estimate_auxiliary import opt_get_model_parameters

from robupy.solve.solve_python import pyth_solve
from robupy.evaluate.evaluate_python import pyth_evaluate

def pyth_criterion(x, data_array, edu_max, delta, edu_start, is_debug,
        is_interpolated, level, measure, min_idx, num_draws_emax, num_periods,
        num_points, is_ambiguous, periods_draws_emax, is_deterministic,
        is_myopic, num_agents, num_draws_prob, periods_draws_prob):
    """ This function provides the wrapper for optimization routines.
    """
    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        opt_get_model_parameters(x, is_debug)


    solution = pyth_solve(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, is_deterministic, is_interpolated, num_draws_emax,
        is_ambiguous, num_periods, num_points, edu_start, is_myopic, is_debug,
        measure, edu_max, min_idx, delta, level,
        shocks_cholesky, periods_draws_emax)

    periods_payoffs_systematic, _, mapping_state_idx, periods_emax, \
        states_all = solution

    crit_val = pyth_evaluate(periods_payoffs_systematic, mapping_state_idx,
        periods_emax, states_all, shocks_cov, shocks_cholesky,
        is_deterministic, num_periods, edu_start, edu_max, delta,
        data_array,  num_agents, num_draws_prob, periods_draws_prob)

    # Finishing
    return crit_val