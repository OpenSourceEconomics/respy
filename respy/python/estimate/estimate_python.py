from respy.python.solve.solve_auxiliary import pyth_calculate_rewards_systematic
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.python.shared.shared_auxiliary import get_log_likl


def pyth_criterion(x, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max,
        data_array, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob, states_all, states_number_period,
        mapping_state_idx, max_states_period, ambi_spec, type_spec,
        optimizer_options):
    """ This function provides the wrapper for optimization routines.
    """

    optim_paras = dist_optim_paras(x, is_debug)

    # Calculate all systematic rewards
    periods_rewards_systematic = pyth_calculate_rewards_systematic(num_periods,
        states_number_period, states_all, edu_start, max_states_period,
        optim_paras, type_spec)

    periods_emax, opt_ambi_details = pyth_backward_induction(num_periods,
        is_myopic, max_states_period, periods_draws_emax, num_draws_emax,
        states_number_period, periods_rewards_systematic, edu_max, edu_start,
        mapping_state_idx, states_all, is_debug, is_interpolated,
        num_points_interp, ambi_spec, optim_paras, optimizer_options, '',
        False)

    contribs = pyth_contributions(periods_rewards_systematic, mapping_state_idx,
        periods_emax, states_all, data_array, periods_draws_prob, tau,
        edu_start, edu_max, num_periods, num_draws_prob, optim_paras, type_spec)

    crit_val = get_log_likl(contribs)

    return crit_val, opt_ambi_details
