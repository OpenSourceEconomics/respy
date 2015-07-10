""" This module contains the functions related to the incorporation of
    ambiguity in the model.
"""

# standard library
from scipy.optimize import minimize
import numpy as np

# project library
from robupy.checks._checks_ambiguity import _checks
import robupy.fort.performance as perf

# module wide variables
HUGE_FLOAT = 10e10

''' Public functions
'''


def simulate_emax_ambiguity(num_draws, eps_standard, period, k,
        payoffs_ex_ante, edu_max, edu_start, mapping_state_idx, states_all,
        num_periods, emax, delta, debug, ambiguity_args):
    """ Get worst case
    """
    # Distribute arguments
    ambiguity = ambiguity_args['ambiguity']
    cholesky = ambiguity_args['cholesky']

    # Auxiliary objects
    measure = ambiguity['measure']
    level = ambiguity['level']

    # Initialize options.
    options = dict()
    options['maxiter'] = 100000000

    # Initialize optimization problem.
    x0 = _get_start(debug)

    # Collect arguments
    args = (num_draws, eps_standard, period, k, payoffs_ex_ante, edu_max,
            edu_start, mapping_state_idx, states_all, num_periods, emax,
            cholesky, delta, debug)

    # Run optimization
    if measure == 'absolute':

        bounds = _prep_absolute(ambiguity, debug)

        opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                       bounds=bounds)

    else:

        constraints = _prep_kl(cholesky, level)

        opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                       constraints=constraints)

    # Write result to file
    if debug:
        _write_result(period, k, opt)

    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant = np.dot(cholesky, eps_standard.T).T
    eps_relevant[:, :2] = eps_relevant[:, :2] + opt['x']
    for j in [0, 1]:
        eps_relevant[:, j] = np.exp(eps_relevant[:, j])

    simulated, payoffs_ex_post, future_payoffs = \
        perf.simulate_emax(num_periods, num_draws, period, k, eps_relevant,
            payoffs_ex_ante, edu_max, edu_start, emax, states_all,
            mapping_state_idx, delta)

    # Debugging
    if debug is True:
        _checks('simulate_emax_ambiguity', simulated, opt)

    # Finishing
    return simulated, payoffs_ex_post, future_payoffs

''' Private functions
'''

def _prep_kl(cholesky, level):
    """ Construct Kullback-Leibler constraint for optimization.
    """
    # Construct covariances
    cov = np.dot(cholesky, cholesky.T)

    # Construct constaint
    constraint_divergence = dict()

    constraint_divergence['type'] = 'ineq'

    constraint_divergence['fun'] = _divergence

    constraint_divergence['args'] = (cov, level)

    # Collection.
    constraints = [constraint_divergence, ]

    # Finishing.
    return constraints

def _divergence(x, cov, level):
    """ Calculate the relevant Kullback-Leibler distance of evaluation points
        from center.
    """
    # Construct alternative distribution
    alt_mean = np.zeros(4)
    alt_mean[:2] = x
    alt_cov = cov

    # Construct baseline distribution
    old_mean, old_cov = np.array([0.0, 0.0, 0.0, 0.0]), cov

    # Calculate distance
    comp_a = np.trace(np.dot(np.linalg.inv(old_cov), alt_cov))

    comp_b = np.dot(np.dot(np.transpose(old_mean - alt_mean),
            np.linalg.inv(old_cov)), (old_mean - alt_mean))

    comp_c = np.log(np.linalg.det(alt_cov) / np.linalg.det(old_cov))

    rslt = 0.5 * (comp_a + comp_b - 4 + comp_c)

    # Finishing.
    return level - rslt

def _criterion(x, num_draws, eps_standard, period, k, payoffs_ex_ante, edu_max,
        edu_start, mapping_state_idx, states_all, num_periods, emax,
        true_cholesky, delta, debug):
    """ Simulate expected future value for alternative shock distributions.
    """
    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant = np.dot(true_cholesky, eps_standard.T).T
    eps_relevant[:, :2] = eps_relevant[:, :2] + x
    for j in [0, 1]:
        eps_relevant[:, j] = np.clip(np.exp(eps_relevant[:, j]), 0.0,
                                     HUGE_FLOAT)

    # Simulate the expected future value for a given parametrization.
    simulated, _, _ = perf.simulate_emax(num_periods, num_draws, period, k,
                        eps_relevant, payoffs_ex_ante, edu_max, edu_start,
                        emax, states_all, mapping_state_idx, delta)
    # Debugging
    if debug is True:
        _checks('_criterion', simulated)

    # Finishing
    return simulated

def _write_result(period, k, opt):
    """ Write result of optimization problem to loggging file.
    """

    string = '''{0[0]:>10} {0[1]:10.4f} {0[2]:10.4f}\n\n'''

    file_ = open('ambiguity.robupy.log', 'a')

    file_.write('PERIOD ' + str(period) + '    State ' + str(k) + '\n' +
                    '-------------------\n\n')

    file_.write(string.format(['Result', opt['x'][0], opt['x'][0]]))

    file_.write('    Success ' + str(opt['success']) + '\n')
    file_.write('    Message ' + opt['message'] + '\n\n\n')

    file_.close()

def _get_start(debug):
    """ Get starting values.
    """
    # Get appropriate starting values
    x0 = [0.00, 0.00]

    # Debugging
    if debug is True:
        _checks('_get_start', x0)

    # Finishing
    return x0

def _prep_absolute(ambiguity, debug):
    """ Get bounds.
    """
    # Distribute information
    level = ambiguity['level']

    # Construct appropriate bounds
    bounds = [[-level, level], [-level, level]]

    # Debugging
    if debug is True:
        _checks('_prep_absolute', bounds)

    # Finishing
    return bounds
