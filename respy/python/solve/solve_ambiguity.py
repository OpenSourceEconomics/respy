from scipy.optimize import minimize
import numpy as np

from respy.python.record.record_ambiguity import record_ambiguity
from respy.python.shared.shared_constants import opt_ambi_info
from respy.python.solve.solve_risk import construct_emax_risk


def construct_emax_ambiguity(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta, shocks_cov,
        measure, optim_paras, optimizer_options, file_sim, is_write):
    """ Construct EMAX accounting for a worst case evaluation.
    """
    is_deterministic = (np.count_nonzero(shocks_cov) == 0)

    args = (num_periods, num_draws_emax, period, k, draws_emax_transformed,
        rewards_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, delta)

    if is_deterministic:
        x_shift, div = [0.0, 0.0], 0.0
        is_success, message = True, 'No random variation in shocks.'

    elif measure == 'abs':
        x_shift, div = [-float(optim_paras['level']), -float(optim_paras['level'])], \
                       float(optim_paras['level'])
        is_success, message = True, 'Optimization terminated successfully.'

    elif measure == 'kl':
        x_shift, is_success, message = get_worst_case(num_periods,
            num_draws_emax, period, k, draws_emax_transformed,
            rewards_systematic, edu_max, edu_start, periods_emax, states_all,
            mapping_state_idx, delta, shocks_cov, optim_paras,
            optimizer_options)

        div = float(-(constraint_ambiguity(x_shift, shocks_cov, optim_paras) -
                      optim_paras['level']))

    else:
        raise NotImplementedError

    if is_write:
        record_ambiguity(period, k, x_shift, div, is_success, message, file_sim)

    emax = criterion_ambiguity(x_shift, *args)

    return emax


def get_worst_case(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta, shocks_cov,
        optim_paras, optimizer_options):
    """ Run the optimization.
    """
    # Initialize options.
    options = dict()
    options['maxiter'] = optimizer_options['SCIPY-SLSQP']['maxiter']
    options['ftol'] = optimizer_options['SCIPY-SLSQP']['ftol']
    options['eps'] = optimizer_options['SCIPY-SLSQP']['eps']

    x0 = np.tile(0.0, 2)

    # Construct constraint
    constraint_divergence = dict()
    constraint_divergence['type'] = 'eq'
    constraint_divergence['fun'] = constraint_ambiguity
    constraint_divergence['args'] = (shocks_cov, optim_paras)

    # Collection.
    constraints = [constraint_divergence, ]

    args = (num_periods, num_draws_emax, period, k, draws_emax_transformed,
        rewards_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, delta)

    # Run optimization
    opt = minimize(criterion_ambiguity, x0, args, method='SLSQP',
        options=options, constraints=constraints)

    # Stabilization. If the optimization fails the starting values are
    # used otherwise it happens that the constraint is not satisfied by far
    # at the return values from the interface.
    if not opt['success']:
        opt['x'] = x0

    is_success, message = opt['success'], opt['message']
    x_shift = opt['x'].tolist()

    # Record some information about all worst-case determinations.
    opt_ambi_info[0] += 1
    if is_success:
        opt_ambi_info[1] += 1

    return x_shift, is_success, message


def criterion_ambiguity(x, num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta):
    """ Evaluating the constructed EMAX with the admissible distribution.
    """
    draws_relevant = draws_emax_transformed.copy()
    for i in range(2):
        draws_relevant[:, i] = draws_relevant[:, i] + x[i]

    emax = construct_emax_risk(num_periods, num_draws_emax, period, k,
        draws_relevant, rewards_systematic, edu_max, edu_start, periods_emax,
        states_all, mapping_state_idx, delta)

    return emax


def constraint_ambiguity(x, shocks_cov, optim_paras):
    """ This function provides the constraints for the SLSQP optimization.
    """

    mean_old = np.zeros(4)

    mean_new = np.zeros(4)
    mean_new[:2] = x

    cov_old = shocks_cov
    cov_new = cov_old

    rslt = optim_paras['level'] - kl_divergence(mean_old, cov_old, mean_new,
        cov_new)

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
