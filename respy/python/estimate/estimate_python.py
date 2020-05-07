from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.python.evaluate.evaluate_python import pyth_evaluate
from respy.python.shared.shared_auxiliary import get_log_likl
import pandas as pd
import numpy as np


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

    contribs = pyth_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, *args)

    crit_val = get_log_likl(contribs)


    contribs_df = pd.DataFrame(columns=["value"], data=contribs.reshape(-1, 1))
    contribs_df["id"] = np.repeat(np.arange(num_agents_est), num_periods)
    contribs_df["period"] = np.tile(np.arange(num_periods), num_agents_est)
    contribs_df.set_index(["id", "period"], inplace=True)
    contribs_df.to_pickle("likelihood_contributions.pickle")


    return crit_val
