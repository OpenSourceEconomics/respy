from scipy.optimize import minimize
import numpy as np

from solve_risk import construct_emax_risk


def kl_divergence(mean_old, cov_old, mean_new, cov_new):
    """ Calculate the Kullback-Leibler divergence.
    """

    num_dims = mean_old.shape[0]

    cov_old_inv = np.linalg.inv(cov_old)
    mean_diff = mean_old - mean_new

    comp_a = np.trace(np.dot(cov_old_inv, cov_new))
    comp_b = np.dot(np.dot(np.transpose(mean_diff), cov_old_inv), mean_diff)
    comp_c = np.log(np.linalg.det(cov_old) / np.linalg.det(cov_new))

    rslt = 0.5 * (comp_a + comp_b - num_dims + comp_c)

    # Finishing.
    return rslt


def construct_emax_ambiguity(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta, level):
    """ Construct EMAX accounting for a worst case evaluation.
    """

    theta = level

    emax = get_worst_case(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta, theta)

    return emax


def get_worst_case(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta, theta):
    """ Run the optimization.
    """

    x0 = np.tile(0, 2)

    args = (num_periods, num_draws_emax, period, k, draws_emax_transformed,
            rewards_systematic, edu_max, edu_start, periods_emax,
            states_all, mapping_state_idx, delta)

    bounds = [(-theta, theta), (-theta, theta)]

    rslt = minimize(criterion_ambiguity, x0, args, method='L-BFGS-B', bounds=bounds)

    # Minimize criterion
    emax = rslt['fun']

    return emax


def criterion_ambiguity(x, num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta):
    """ Evaluating the constructed EMAX with the admissible distribution.
    """
    draws_relevant = draws_emax_transformed.copy()
    for i in range(2):
        draws_relevant[:, i] = draws_relevant[:, i] + x[i]

    emax = construct_emax_risk(num_periods, num_draws_emax, period, k,
        draws_relevant, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta)

    return emax
