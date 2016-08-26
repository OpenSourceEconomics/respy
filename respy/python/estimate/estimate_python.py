from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.shared.shared_auxiliary import get_log_likl


def pyth_criterion(x, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max, delta,
        data_array, num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob, states_all, states_number_period,
        mapping_state_idx, max_states_period, level, is_ambiguity):
    """ This function provides the wrapper for optimization routines.
    """

    args = (is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_start, is_debug, edu_max, delta, data_array,
        num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob, states_all, states_number_period,
        mapping_state_idx, max_states_period, level, is_ambiguity)

    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky \
        = dist_optim_paras(x, is_debug)

    contribs = pyth_contributions(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, *args)

    crit_val = get_log_likl(contribs)

    return crit_val
