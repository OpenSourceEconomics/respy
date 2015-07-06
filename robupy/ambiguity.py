""" This module contains the functions related to the incorporation of
    ambiguity in the model.
"""

# standard library
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# project library
from robupy.checks._checks_ambiguity import _checks
from robupy.shared import *


''' Public functions
'''


def simulate_emax_ambiguity(num_draws, eps_standard,
        period, k, payoffs_ex_ante, edu_max, edu_start,
        mapping_state_idx, states_all, num_periods, emax,
        delta, debug, ambiguity_args):
    """ Get worst case
    """
    # Distribute arguments
    ambiguity = ambiguity_args['with_ambiguity']
    cholesky = ambiguity_args['cholesky']

    # Initialize options.
    options = dict()

    options['maxiter'] = 10000

    # Initialize optimization problem.
    x0 = _get_start(ambiguity, debug)

    bounds = _get_bounds(ambiguity, debug)

    # Collect arguments
    args = (num_draws, eps_standard, period,
            k, payoffs_ex_ante, edu_max, edu_start,
            mapping_state_idx, states_all,
            num_periods, emax, cholesky, delta, debug)

    # Run optimization
    opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                   bounds=bounds)

    # Write result to file
    if debug:
        _write_result(period, k, opt)

    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant = np.dot(cholesky, eps_standard.T).T + opt['x']
    for j in [0, 1]:
        eps_relevant[:, j] = np.exp(eps_relevant[:, j])

    simulated, payoffs_ex_post, future_payoffs = simulate_emax(num_draws,
        period, k, eps_relevant, payoffs_ex_ante,
        edu_max, edu_start, num_periods, emax, states_all,
        mapping_state_idx, delta)

    # Debugging
    if debug is True:
        _checks('simulate_emax_ambiguity', simulated, opt)

    # Finishing
    return simulated, payoffs_ex_post, future_payoffs

''' Private functions
'''


def _criterion(x, num_draws, eps_standard, period,
                   k, payoffs_ex_ante, edu_max, edu_start,
                   mapping_state_idx, states_all,
                   num_periods, emax, true_cholesky, delta, debug):
    """ Simulate expected future value for alternative shock distributions.
    """
    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant = np.dot(true_cholesky, eps_standard.T).T + x
    for j in [0, 1]:
        eps_relevant[:, j] = np.exp(eps_relevant[:, j])

    # Simulate the expected future value for a given parameterization.
    simulated, _, _ = \
        simulate_emax(num_draws, period, k,
                      eps_relevant, payoffs_ex_ante, edu_max,
                      edu_start, num_periods, emax, states_all,
                      mapping_state_idx, delta)

    # Debugging
    if debug is True:
        _checks('_criterion', simulated)

    # Finishing
    return simulated

def _write_result(period, k, opt):
    """ Write result of optimization problem to loggging file.
    """

    file_ = open('ambiguity.robupy.log', 'a')

    file_.write('Period ' + str(period) + '    State ' + str(k) + '\n' +
                    '-------------------\n\n')

    file_.write('Result  ' + str(opt['x']) + '\n')
    file_.write('Success ' + str(opt['success']) + '\n')
    file_.write('Message ' + opt['message'] + '\n\n\n')

    file_.close()

def _get_start(ambiguity, debug):
    """ Get starting values.
    """
    # Distribute information
    measure = ambiguity['measure']

    # Get appropriate starting values
    if measure == 'absolute':
        x0 = [0.00, 0.00, 0.00, 0.00]

    # Debugging
    if debug is True:
        _checks('_get_start', x0, measure)

    # Finishing
    return x0


def _get_bounds(ambiguity, debug):
    """ Get bounds.
    """
    # Distribute information
    measure = ambiguity['measure']
    level = ambiguity['level']

    # Construct appropriate bounds
    if measure == 'absolute':
        bounds = [[-level, level], [-level, level],
                  [-level, level], [-level, level]]

    # Debugging
    if debug is True:
        _checks('_get_bounds', bounds, measure)

    # Finishing
    return bounds


