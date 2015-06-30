""" This module contains the functions related to the incorporation of
    ambiguity in the model.
"""

# standard library
import numpy as np
from scipy.optimize import minimize

# project library
from robupy.checks._checks_ambiguity import _checks
from robupy._shared import _get_future_payoffs


''' Public functions
'''


def simulate_emax_ambiguity(num_draws, period_payoffs_ex_post, eps_standard,
        period, k, period_payoffs_ex_ante, edu_max, edu_start,
        mapping_state_idx, states_all, future_payoffs, num_periods, emax,
        ambiguity, true_cholesky):
    """ Get worst case
    """

    # Initialize options.
    options = dict()

    options['maxiter'] = 10000

    # Initialize optimization problem.
    x0 = _get_start(ambiguity)

    bounds = _get_bounds(ambiguity)

    # Collect arguments
    args = (num_draws, period_payoffs_ex_post, eps_standard, period,
            k, period_payoffs_ex_ante, edu_max, edu_start,
            mapping_state_idx, states_all, future_payoffs,
            num_periods, emax, true_cholesky)

    # Run optimization
    opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                   bounds=bounds)

    # Extract information
    fun = opt['fun']

    # Finishing
    return fun


''' Private functions
'''


def _criterion(x, num_draws, period_payoffs_ex_post, eps_standard, period,
                   k, period_payoffs_ex_ante, edu_max, edu_start,
                   mapping_state_idx, states_all, future_payoffs,
                   num_periods, emax, true_cholesky):
    """ Simulate expected future value for alternative shock distributions.
    """

    # Transformations
    eps_relevant = np.dot(true_cholesky, eps_standard.T).T + x

    for j in [0, 1]:
        eps_relevant[:, j] = np.exp(eps_relevant[:, j])

    # Initialize container
    simulated = 0.0

    # Calculate maximum value
    for i in range(num_draws):

        # Calculate ex post payoffs
        for j in [0, 1]:
            period_payoffs_ex_post[period, k, j] = \
                period_payoffs_ex_ante[period, k, j] * \
                eps_relevant[i, j]

        for j in [2, 3]:
            period_payoffs_ex_post[period, k, j] = \
                period_payoffs_ex_ante[period, k, j] + \
                eps_relevant[i, j]

        # Check applicability
        if period == (num_periods - 1):
            continue

        # Get future values
        future_payoffs[period, k, :] = \
            _get_future_payoffs(edu_max, edu_start, mapping_state_idx,
                                period, emax, k, states_all)

        # Calculate total utilities
        total_payoffs = period_payoffs_ex_post[period, k, :] + \
            future_payoffs[period, k, :]

        # Determine optimal choice
        maximum = max(total_payoffs)

        # Recording expected future value
        simulated += maximum

    # Scaling
    simulated = simulated / num_draws

    # Finishing
    return simulated



def _get_start(ambiguity):
    """ Get starting value.
    """
    # Distribute information
    measure = ambiguity['measure']
    level = ambiguity['level']
    para = ambiguity['para']

    if measure == 'absolute':

        x0 = [0.00, 0.00, 0.00, 0.00]

    return x0


def _get_bounds(ambiguity):
    """ Get bounds.
    """
    # Distribute information
    measure = ambiguity['measure']
    level = ambiguity['level']
    para = ambiguity['para']

    if measure == 'absolute':

        bounds = [[-level, level], [-level, level],
                      [-level, level], [-level, level]]

    return bounds

