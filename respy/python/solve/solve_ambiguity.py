from scipy.optimize import minimize
import numpy as np

from respy.python.solve.solve_risk import construct_emax_risk


def construct_emax_ambiguity(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta, shocks_cov, level):
    """ Construct EMAX accounting for a worst case evaluation.
    """

    emax = get_worst_case(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta, shocks_cov, level)

    return emax


def get_worst_case(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta, shocks_cov, level):
    """ Run the optimization.
    """
    # Initialize options.
    options = dict()
    options['maxiter'] = 100000000

    x0 = np.tile(0.0, 2)

    # Construct constraint
    constraint_divergence = dict()
    constraint_divergence['type'] = 'eq'
    constraint_divergence['fun'] = constraint_ambiguity
    constraint_divergence['args'] = (shocks_cov, level)

    # Collection.
    constraints = [constraint_divergence, ]

    args = (num_periods, num_draws_emax, period, k, draws_emax_transformed,
        rewards_systematic, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, delta)

    # Run optimization
 #   opt = minimize(criterion_ambiguity, x0, args, method='SLSQP',
 #       options=options, constraints=constraints)

    # Stabilization. If the optimization fails the starting values are
    # used otherwise it happens that the constraint is not satisfied by far
    # at the return values from the interface.
 #   if not opt['success']:
 #       opt['x'] = x0

    # TODO: Channel debug, write
#    if True:
#        div = constraint_ambiguity(opt['x'], shocks_cov, level)
#        write_result(period, k, opt, div)
    # Logging result to file
 #   if is_debug:
        # Evaluate divergence at final value.
#        div = divergence(opt['x'], shocks_cov, level) - level
#        _write_result(period, k, opt, div)


 #   x_shift = opt['x']

    x_shift = [-level, -level]

    emax = criterion_ambiguity(x_shift, *args)

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
        draws_relevant, rewards_systematic, edu_max, edu_start, periods_emax,
        states_all, mapping_state_idx, delta)

    return emax


def constraint_ambiguity(x, shocks_cov, level):
    """ This function provides the constraints for the SLSQP optimization.
    """

    mean_old = np.zeros(4)

    mean_new = np.zeros(4)
    mean_new[:2] = x

    cov_old = shocks_cov
    cov_new = cov_old

    rslt = level - kl_divergence(mean_old, cov_old, mean_new, cov_new)

    return rslt


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

def write_result(period, k, opt, div):
    """ Write result of optimization problem to loggging file.
    """

    with open('amb.respy.log', 'a') as file_:

        string = ' PERIOD{0[0]:>7}  STATE{0[1]:>7}\n\n'
        file_.write(string.format([period, k]))

        string = '    {0[0]:<13} {0[1]:10.4f} {0[2]:10.4f}\n'
        file_.write(string.format(['Result', opt['x'][0], opt['x'][1]]))
        string = '    {0[0]:<13} {0[1]:10.4f}\n\n'
        file_.write(string.format(['Divergence', div]))

        file_.write('    Success ' + str(opt['success']) + '\n')
        file_.write('    Message ' + opt['message'] + '\n\n')
