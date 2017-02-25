from scipy.optimize import minimize
import numpy as np

from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.solve.solve_risk import construct_emax_risk


def construct_emax_ambiguity(num_periods, num_draws_emax, period, k,
        draws_emax_standard, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, shocks_cov, measure, mean,
        optim_paras, optimizer_options, opt_ambi_details):
    """ Construct EMAX accounting for a worst case evaluation.
    """
    is_deterministic = (np.count_nonzero(shocks_cov) == 0)

    base_args = (num_periods, num_draws_emax, period, k,
                 draws_emax_standard,
        rewards_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx)

    shocks_cov = np.matmul(optim_paras['shocks_cholesky'],
        optim_paras['shocks_cholesky'].T)


    if is_deterministic:
        x_shift, div = [0.0, 0.0, 0.0, 0.0, 0.0], 0.0
        is_success, mode = True, 15

    elif measure == 'abs':
        #TODO: REvisit later.
        x_shift, div = [-float(optim_paras['level']), -float(optim_paras['level'])], \
                       float(optim_paras['level'])
        is_success, mode = True, 16

    elif measure == 'kl':

        args = ()
        args += base_args + (shocks_cov, optim_paras, optimizer_options, mean)
        x_shift, is_success, mode = get_worst_case(*args)

        div = float(-(constraint_ambiguity(x_shift, shocks_cov, optim_paras) -
                      optim_paras['level']))

    else:
        raise NotImplementedError

    # We collect the information from the optimization step for future
    # recording.
    args = ()
    args += (x_shift[0], x_shift[1], div, float(is_success), mode)
    opt_ambi_details[period, k, :] = args

    args = ()
    args += base_args + (optim_paras,)
    emax = criterion_ambiguity(x_shift, *args)

    return emax, opt_ambi_details


def get_worst_case(num_periods, num_draws_emax, period, k, draws_emax_standard,
        rewards_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, shocks_cov, optim_paras, optimizer_options, mean):
    """ Run the optimization.
    """
    # Initialize options.
    options = dict()
    options['maxiter'] = optimizer_options['SCIPY-SLSQP']['maxiter']
    options['ftol'] = optimizer_options['SCIPY-SLSQP']['ftol']
    options['eps'] = optimizer_options['SCIPY-SLSQP']['eps']

    if mean:
        x0 = np.tile(0.0, 2)
    else:
        x0 = np.tile(0.0, 5)

    # Construct constraint
    constraint_divergence = dict()
    constraint_divergence['type'] = 'eq'
    constraint_divergence['fun'] = constraint_ambiguity
    constraint_divergence['args'] = (shocks_cov, optim_paras)

    # Collection.
    constraints = [constraint_divergence, ]

    args = (num_periods, num_draws_emax, period, k, draws_emax_standard,
        rewards_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, optim_paras)

    # Run optimization
    opt = minimize(criterion_ambiguity, x0, args, method='SLSQP',
        options=options, constraints=constraints)

    # Stabilization. If the optimization fails the starting values are
    # used otherwise it happens that the constraint is not satisfied by far
    # at the return values from the interface.
    if not opt['success']:
        opt['x'] = x0

    is_success, mode = opt['success'], opt['status']
    x_shift = opt['x'].tolist()

    return x_shift, is_success, mode


def criterion_ambiguity(x, num_periods, num_draws_emax, period, k,
        draws_emax_standard, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, optim_paras):
    """ Evaluating the constructed EMAX with the admissible distribution.
    """

    mean = np.append(x[:2], [0.0, 0.0])

    draws_emax_relevant = transform_disturbances(draws_emax_standard,
        mean, optim_paras['shocks_cholesky'])

    emax = construct_emax_risk(num_periods, num_draws_emax, period, k,
        draws_emax_relevant, rewards_systematic, edu_max, edu_start, periods_emax,
        states_all, mapping_state_idx, optim_paras)

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
