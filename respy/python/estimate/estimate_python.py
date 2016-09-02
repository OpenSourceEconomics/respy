from respy.python.solve.solve_auxiliary import pyth_calculate_rewards_systematic
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.python.shared.shared_auxiliary import get_log_likl


def pyth_criterion(x, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max, delta,
        data_array, num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob, states_all, states_number_period,
        mapping_state_idx, max_states_period, is_ambiguity, measure, level):
    """ This function provides the wrapper for optimization routines.
    """

    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky \
        = dist_optim_paras(x, is_debug)

    # Calculate all systematic rewards
    periods_rewards_systematic = pyth_calculate_rewards_systematic(num_periods,
        states_number_period, states_all, edu_start, coeffs_a, coeffs_b,
        coeffs_edu, coeffs_home, max_states_period)

    periods_emax = pyth_backward_induction(num_periods, is_myopic,
        max_states_period, periods_draws_emax, num_draws_emax,
        states_number_period, periods_rewards_systematic, edu_max, edu_start,
        mapping_state_idx, states_all, delta, is_debug, is_interpolated,
        num_points_interp, shocks_cholesky, is_ambiguity, measure, level,
        False)

    contribs = pyth_contributions(periods_rewards_systematic, mapping_state_idx,
        periods_emax, states_all, shocks_cholesky, data_array,
        periods_draws_prob, delta, tau, edu_start, edu_max, num_agents_est,
        num_periods, num_draws_prob)

    crit_val = get_log_likl(contribs)

    return crit_val
