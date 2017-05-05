from scipy.optimize import minimize
import numpy as np

from respy.python.shared.shared_auxiliary import covariance_to_correlation
from respy.python.shared.shared_auxiliary import correlation_to_covariance
from respy.python.solve.solve_risk import construct_emax_risk
from respy.python.shared.shared_constants import SMALL_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT


def construct_emax_ambiguity(num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard,
        draws_emax_ambiguity_transformed, rewards_systematic, periods_emax, states_all,
        mapping_state_idx, edu_spec, ambi_spec, optim_paras, optimizer_options, opt_ambi_details):
    """ Construct EMAX accounting for a worst case evaluation.
    """
    # Construct auxiliary objects
    shocks_cov = np.matmul(optim_paras['shocks_cholesky'], optim_paras['shocks_cholesky'].T)

    # Determine special cases
    is_deterministic = (np.count_nonzero(shocks_cov) == 0)

    base_args = (num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard,
                 draws_emax_ambiguity_transformed, rewards_systematic, periods_emax, states_all,
                 mapping_state_idx)

    # The following two scenarios are only maintained for testing and debugging purposes.
    if is_deterministic:
        rslt_mean, rslt_sd = [0.0, 0.0], [0.0, 0.0]
        div, is_success, mode = 0.0, 1.0, 15

    elif ambi_spec['measure'] == 'abs':
        rslt_mean = [-optim_paras['level'], -optim_paras['level']]
        rslt_sd = np.sqrt(shocks_cov[[(0, 1), (0, 1)]])
        div, is_success, mode = optim_paras['level'], 1.0, 16

    elif ambi_spec['measure'] == 'kl':
        # In conflict with the usual design, we pass in shocks_cov directly. Otherwise it needs
        # to be constructed over and over for each of the evaluations of the criterion functions.
        args = ()
        args += base_args + (shocks_cov, edu_spec, optim_paras, optimizer_options)
        args += (ambi_spec, )
        opt_return, is_success, mode = get_worst_case(*args)

        # We construct the complete results depending on the actual request.
        rslt_mean = opt_return[:2]
        if ambi_spec['mean']:
            rslt_sd = np.sqrt(shocks_cov[[(0, 1), (0, 1)]])
        else:
            rslt_sd = opt_return[2:]

        args = ()
        args += (opt_return, shocks_cov, optim_paras)
        div = -(constraint_ambiguity(*args) - optim_paras['level'])

    else:
        raise NotImplementedError

    # Now we recombine the results from the optimization for easier access.
    rslt_all = np.append(rslt_mean, rslt_sd)

    # We collect the information from the optimization step for future recording.
    opt_ambi_details[period, k, :4] = rslt_all
    opt_ambi_details[period, k, 4:] = (div, is_success, mode)

    # The optimizer does not always return the actual value of the criterion function at the
    # optimum.
    args = ()
    args += base_args + (edu_spec, optim_paras, shocks_cov, ambi_spec)
    emax = criterion_ambiguity(rslt_all, *args)

    return emax, opt_ambi_details


def get_worst_case(num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard,
        draws_emax_ambiguity_transformed, rewards_systematic, periods_emax, states_all,
        mapping_state_idx, shocks_cov, edu_spec, optim_paras, optimizer_options, ambi_spec):
    """ Run the optimization.
    """
    # Initialize options.
    options = dict()
    options['maxiter'] = optimizer_options['SCIPY-SLSQP']['maxiter']
    options['ftol'] = optimizer_options['SCIPY-SLSQP']['ftol']
    options['eps'] = optimizer_options['SCIPY-SLSQP']['eps']

    # The construction of starting value is straightforward. It is simply the benchmark model.
    sd_base = np.sqrt(shocks_cov[[(0, 1), (0, 1)]])
    mean_base = np.tile(0.0, 2)

    x0 = mean_base
    if not ambi_spec['mean']:
        x0 = np.append(x0, sd_base)

    # Construct constraint
    constraint_divergence = dict()
    constraint_divergence['type'] = 'eq'
    constraint_divergence['fun'] = constraint_ambiguity
    constraint_divergence['args'] = (shocks_cov, optim_paras)
    constraint_divergence['args'] = (shocks_cov, optim_paras)

    # Construct bounds.
    bounds = []
    bounds += [[None, None]]
    bounds += [[None, None]]
    if not ambi_spec['mean']:
        bounds += [[0.00 + SMALL_FLOAT, None]]
        bounds += [[0.00 + SMALL_FLOAT, None]]

    # Collection.
    constraints = [constraint_divergence, ]

    args = (num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard,
            draws_emax_ambiguity_transformed, rewards_systematic, periods_emax, states_all,
            mapping_state_idx, edu_spec, optim_paras, shocks_cov, ambi_spec)

    # Run optimization
    opt = minimize(criterion_ambiguity, x0, args, method='SLSQP', options=options,
        constraints=constraints, bounds=bounds)

    # Stabilization. If the optimization fails the starting values are used otherwise it happens
    # that the constraint is not satisfied by far at the return values from the interface.
    if not opt['success']:
        opt['x'] = x0

    is_success, mode = float(opt['success']), opt['status']
    rslt = opt['x'].tolist()

    return rslt, is_success, mode


def criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard,
        draws_emax_ambiguity_transformed, rewards_systematic, periods_emax, states_all,
        mapping_state_idx, edu_spec, optim_paras, shocks_cov, ambi_spec):
    """ Evaluating the constructed EMAX with the admissible distribution.
    """
    # We construct the candidate values for the mean and the dependence of the shocks.
    shocks_cholesky_cand = get_relevant_dependence(shocks_cov, x)[1]
    shocks_mean_cand = np.append(x[:2], [0.0, 0.0])

    if ambi_spec['mean']:
        draws_emax_relevant = draws_emax_ambiguity_transformed.copy()
    else:
        draws_emax_relevant = np.dot(shocks_cholesky_cand, draws_emax_ambiguity_standard.T).T

    for i in range(2):
        draws_emax_relevant[:, i] += shocks_mean_cand[i]

    for i in range(2):
        draws_emax_relevant[:, i] = np.clip(np.exp(draws_emax_relevant[:, i]),
            0.0, HUGE_FLOAT)

    emax = construct_emax_risk(num_periods, num_draws_emax, period, k, draws_emax_relevant,
        rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec, optim_paras)

    return emax


def get_relevant_dependence(shocks_cov, x):
    """ This function creates the objects that describe the relevant dependence structures for 
    the random disturbances during the worst-case determination.
    """
    # We need to deal with the special case when there is no variation in the random disturbances.
    is_deterministic = (np.count_nonzero(shocks_cov) == 0)
    if is_deterministic:
        return np.zeros((4, 4)), np.zeros((4, 4))

    if len(x) == 2:
        shocks_cholesky_cand = np.linalg.cholesky(shocks_cov)
        shocks_cov_cand = shocks_cov
    else:
        # Update the correlation matrix
        shocks_corr_base = covariance_to_correlation(shocks_cov)
        sd = np.append(x[2:], np.sqrt(shocks_cov[(2, 3), (2, 3)]))
        shocks_cov_cand = correlation_to_covariance(shocks_corr_base, sd)
        shocks_cholesky_cand = np.linalg.cholesky(shocks_cov_cand)

    return shocks_cov_cand, shocks_cholesky_cand


def constraint_ambiguity(x, shocks_cov, optim_paras):
    """ This function provides the constraints for the SLSQP optimization.
    """
    # Construct the means of the two candidate distributions.
    mean_new = np.array(np.append(x[:2], np.zeros(2)))
    mean_old = np.zeros(4)

    # Construct the two covariances of the two candidate distributions.
    cov_old = shocks_cov
    cov_new = get_relevant_dependence(shocks_cov, x)[0]

    # Evaluate the constraint for the SLSQP algorithm.
    args = ()
    args += (mean_old, cov_old, mean_new, cov_new)
    rslt = optim_paras['level'] - kl_divergence(*args)

    return rslt


def kl_divergence(mean_old, cov_old, mean_new, cov_new):
    """ Calculate the Kullback-Leibler divergence.
    """

    num_dims = mean_old.shape[0]

    cov_old_inv = np.linalg.pinv(cov_old)
    mean_diff = mean_old - mean_new

    comp_a = np.trace(np.dot(cov_old_inv, cov_new))
    comp_b = np.dot(np.dot(np.transpose(mean_diff), cov_old_inv), mean_diff)
    comp_c = np.log(np.linalg.det(cov_old) / np.linalg.det(cov_new))

    rslt = 0.5 * (comp_a + comp_b - num_dims + comp_c)

    # Finishing.
    return rslt
