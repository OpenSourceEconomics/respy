from scipy.optimize import minimize
import numpy as np

from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.solve.solve_risk import construct_emax_risk


def construct_emax_ambiguity(num_periods, num_draws_emax, period, k,
        draws_emax_standard, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, shocks_cov, ambi_spec,
        optim_paras, optimizer_options, opt_ambi_details):
    """ Construct EMAX accounting for a worst case evaluation.
    """
    is_deterministic = (np.count_nonzero(shocks_cov) == 0)

    base_args = (num_periods, num_draws_emax, period, k,
        draws_emax_standard, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx)

    # The following two senarios are only maintained for testing and
    # debugging purposes.
    if is_deterministic:
        ambi_rslt_mean_subset = [0.0, 0.0]
        ambi_rslt_chol_subset = [0.0, 0.0, 0.0]
        div, is_success, mode = 0.0, True, 15

    elif ambi_spec['measure'] == 'abs':
        ambi_rslt_mean_subset = [-optim_paras['level'], -optim_paras['level']]
        ambi_rslt_chol_subset = get_upper_cholesky(optim_paras)
        div, is_success, mode = optim_paras['level'], True, 16

    elif ambi_spec['measure'] == 'kl':

        args = ()
        args += base_args + (shocks_cov, optim_paras, optimizer_options)
        args += (ambi_spec, )
        ambi_rslt_return, is_success, mode = get_worst_case(*args)

        # We construct the complete results depending on the actual request.
        ambi_rslt_mean_subset = ambi_rslt_return[:2]
        if ambi_spec['mean']:
            ambi_rslt_chol_subset = get_upper_cholesky(optim_paras)
        else:
            ambi_rslt_chol_subset = ambi_rslt_return[2:]

        args = ()
        args += (ambi_rslt_return, shocks_cov, optim_paras)
        div = -(constraint_ambiguity(*args) - optim_paras['level'])

    else:
        raise NotImplementedError

    # Now we recombine the results from the optimization for easier access.
    ambi_rslt_all = np.append(ambi_rslt_mean_subset, ambi_rslt_chol_subset)

    # We collect the information from the optimization step for future
    # recording.
    opt_ambi_details[period, k, :5] = ambi_rslt_all
    opt_ambi_details[period, k, 5:] = (div, is_success, mode)

    # The optimizer sdoes not always return the actual value of the criterion
    # function at the optimium.
    args = ()
    args += base_args + (optim_paras, shocks_cov)
    emax = criterion_ambiguity(ambi_rslt_all, *args)

    return emax, opt_ambi_details


def get_worst_case(num_periods, num_draws_emax, period, k, draws_emax_standard,
        rewards_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, shocks_cov, optim_paras, optimizer_options,
        ambi_spec):
    """ Run the optimization.
    """
    # Initialize options.
    options = dict()
    options['maxiter'] = optimizer_options['SCIPY-SLSQP']['maxiter']
    options['ftol'] = optimizer_options['SCIPY-SLSQP']['ftol']
    options['eps'] = optimizer_options['SCIPY-SLSQP']['eps']

    # The construction of starting value is straightforward. It is simply the
    # benchmark model.
    x0 = np.tile(0.0, 2)
    if not ambi_spec['mean']:
        x_chol = get_upper_cholesky(optim_paras)
        x0 = np.append(x0, x_chol)

    # Construct constraint
    constraint_divergence = dict()
    constraint_divergence['type'] = 'eq'
    constraint_divergence['fun'] = constraint_ambiguity
    constraint_divergence['args'] = (shocks_cov, optim_paras)

    # Collection.
    constraints = [constraint_divergence, ]

    args = (num_periods, num_draws_emax, period, k, draws_emax_standard,
        rewards_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, optim_paras, shocks_cov)

    # Run optimization
    opt = minimize(criterion_ambiguity, x0, args, method='SLSQP',
        options=options, constraints=constraints)

    # Stabilization. If the optimization fails the starting values are
    # used otherwise it happens that the constraint is not satisfied by far
    # at the return values from the interface.
    if not opt['success']:
        opt['x'] = x0

    is_success, mode = float(opt['success']), opt['status']
    ambi_rslt_return = opt['x'].tolist()

    return ambi_rslt_return, is_success, mode


def criterion_ambiguity(x, num_periods, num_draws_emax, period, k,
        draws_emax_standard, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, optim_paras, shocks_cov):
    """ Evaluating the constructed EMAX with the admissible distribution.
    """
    # First we construct the relevant mean.
    ambi_cand_mean_subset = x[:2]
    ambi_cand_mean_full = np.append(ambi_cand_mean_subset, [0.0, 0.0])

    # Now we turn to the more complex construction of the relevant Cholesky
    # decomposition.
    if len(x) == 2:
        # This is the case where there is only ambiguity about the mean
        # values.
        ambi_cand_cholesky = optim_paras['shocks_cholesky'].copy()
    elif len(x) == 5:
        ambi_cand_chol_subset = x[2:]
        _, ambi_cand_cholesky = construct_full_covariances(ambi_cand_chol_subset,
            shocks_cov)
    else:
        raise AssertionError

    # Now we can construct the relevant random draws from the standard deviates.
    draws_emax_relevant = transform_disturbances(draws_emax_standard,
        ambi_cand_mean_full, ambi_cand_cholesky)

    emax = construct_emax_risk(num_periods, num_draws_emax, period, k,
        draws_emax_relevant, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, optim_paras)

    return emax


def get_upper_cholesky(optim_paras):
    """ Extract the upper 2 x 2 block of the Cholesky decomposition.
    """
    return optim_paras['shocks_cholesky'][:2, :2][np.tril_indices(2)].copy()


def construct_full_covariances(ambi_cand_chol_flat, shocks_cov):
    """ We determine the worst-case Cholesky factors so we need to construct
    the full set of factors.
    """
    ambi_cand_chol_subset = np.zeros((2, 2))
    ambi_cand_chol_subset[np.triu_indices(2)] = ambi_cand_chol_flat

    args = (ambi_cand_chol_subset, ambi_cand_chol_subset.T)
    ambi_cand_cov_subset = np.matmul(*args)

    ambi_cand_cov = shocks_cov.copy()
    ambi_cand_cov[:2, :2] = ambi_cand_cov_subset

    ambi_cand_cho = np.linalg.cholesky(ambi_cand_cov)

    return ambi_cand_cov, ambi_cand_cho


def constraint_ambiguity(x, shocks_cov, optim_paras):
    """ This function provides the constraints for the SLSQP optimization.
    """

    # Construct the means of the two candidate distributions.
    mean_new = np.array(np.append(x[:2], np.zeros(2)))
    mean_old = np.zeros(4)

    # Construct the two covariances of the two candidate distributions.
    cov_old = shocks_cov
    if len(x) == 5:
        cov_new, _ = construct_full_covariances(x[2:], shocks_cov)
    else:
        cov_new = cov_old

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
