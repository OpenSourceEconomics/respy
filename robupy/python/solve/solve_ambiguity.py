""" This module contains the functions related to the incorporation of
    ambiguity in the model.
"""

# standard library
import numpy as np
from scipy.optimize import minimize

# project library
from robupy.python.solve.solve_emax import simulate_emax

''' Public functions
'''


def get_payoffs_ambiguity(num_draws_emax, draws_emax, period, k,
        payoffs_systematic, edu_max, edu_start, mapping_state_idx, states_all,
        num_periods, periods_emax, delta, is_debug, shocks_cov, level,
        is_deterministic, shocks_cholesky):
    """ Determine the worst case payoffs.
    """
    # Determine the worst case, special attention to zero variability. The
    # latter is included as a special case for debugging purposes. The worst
    # case corresponds to zero.
    if is_deterministic:
        opt = handle_shocks_zero(is_debug, period, k)
    else:
        opt = get_worst_case(num_draws_emax, draws_emax,
            period, k, payoffs_systematic, edu_max, edu_start,
            mapping_state_idx, states_all, num_periods, periods_emax,
            delta, is_debug, shocks_cov, level, shocks_cholesky)

    # Simulate the expected future value for the worst case outcome
    simulated = simulate_emax(num_periods, num_draws_emax, period, k,
        draws_emax, payoffs_systematic, edu_max, edu_start, periods_emax,
        states_all, mapping_state_idx, delta, shocks_cholesky, opt['x'])

    # Debugging. This only works in the case of success, as otherwise
    # opt['fun'] is not equivalent to simulated.
    if is_debug and opt['success']:
        _checks_ambiguity('get_payoffs_ambiguity', simulated, opt)

    # Finishing
    return simulated

''' Auxiliary functions
'''


def handle_shocks_zero(is_debug, period, k):
    """ This function ensures that the special case of zero variability is
    handled with care. This is used for debugging purposes.
    """
    # Set up mock object.
    opt = dict()

    opt['message'] = 'No random variation in shocks.'
    opt['success'] = False
    opt['x'] = np.zeros(2)
    opt['fun'] = 0.0

    # Write information to log file.
    if is_debug:
        _write_result(period, k, opt, 0.0)

    # Finishing
    return opt


def get_worst_case(num_draws_emax, draws_emax, period, k,
        payoffs_systematic, edu_max, edu_start, mapping_state_idx, states_all,
        num_periods, periods_emax, delta, is_debug, shocks_cov, level,
        shocks_cholesky):
    """ Determine the worst case outcome for the given parametrization.
    """
    # Initialize options.
    options = dict()
    options['maxiter'] = 100000000

    # Initialize optimization problem.
    x0 = _get_start(is_debug)

    # Collect arguments
    args = (num_draws_emax, draws_emax, period, k, payoffs_systematic,
            edu_max, edu_start, mapping_state_idx, states_all, num_periods,
            periods_emax, delta, shocks_cholesky)

    # Run optimization
    constraints = _prep_kl(shocks_cov, level)
    opt = minimize(criterion_ambiguity, x0, args, method='SLSQP',
        options=options, constraints=constraints)
    # Stabilization. If the optimization fails the starting values are
    # used otherwise it happens that the constraint is not satisfied by far.
    if not opt['success']:
        opt['x'] = x0

    # Logging result to file
    if is_debug:
        # Evaluate divergence at final value.
        div = divergence(opt['x'], shocks_cov, level) - level
        _write_result(period, k, opt, div)

    # Finishing
    return opt


def criterion_ambiguity(x, num_draws_emax, draws_emax, period, k,
        payoffs_systematic, edu_max, edu_start, mapping_state_idx,
        states_all, num_periods, periods_emax, delta, shocks_cholesky):
    """ Simulate expected future value for alternative shock distributions.
    """
    # Simulate the expected future value for a given parametrization.
    simulated = simulate_emax(num_periods, num_draws_emax, period, k,
        draws_emax, payoffs_systematic, edu_max, edu_start, periods_emax,
        states_all, mapping_state_idx, delta, shocks_cholesky, x)

    # Debugging
    _checks_ambiguity('criterion_ambiguity', simulated)

    # Finishing
    return simulated


def divergence(x, cov, level):
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


def _prep_kl(shocks_cov, level):
    """ Construct Kullback-Leibler constraint for optimization.
    """
    # Construct constraint
    constraint_divergence = dict()
    constraint_divergence['type'] = 'eq'
    constraint_divergence['fun'] = divergence
    constraint_divergence['args'] = (shocks_cov, level)

    # Collection.
    constraints = [constraint_divergence, ]

    # Finishing.
    return constraints


def _write_result(period, k, opt, div):
    """ Write result of optimization problem to loggging file.
    """

    with open('ambiguity.robupy.log', 'a') as file_:

        string = ' PERIOD{0[0]:>7}  STATE{0[1]:>7}\n\n'
        file_.write(string.format([period, k]))

        string = '    {0[0]:<13} {0[1]:10.4f} {0[2]:10.4f}\n'
        file_.write(string.format(['Result', opt['x'][0], opt['x'][1]]))
        string = '    {0[0]:<13} {0[1]:10.4f}\n\n'
        file_.write(string.format(['Divergence', div]))

        file_.write('    Success ' + str(opt['success']) + '\n')
        file_.write('    Message ' + opt['message'] + '\n\n')


def _get_start(is_debug):
    """ Get starting values.
    """
    # Get appropriate starting values
    x0 = [0.00, 0.00]

    # Debugging
    if is_debug:
        _checks_ambiguity('_get_start', x0)

    # Finishing
    return x0


def _checks_ambiguity(str_, *args):
    """ This checks the integrity of the objects related to the
        solution of the model.
    """

    if str_ == '_get_start':

        # Distribute input parameters
        x0, = args

        # Check quality of starting values
        assert (len(x0) == 2)
        assert (np.all(np.isfinite(x0)))

        assert (all(val == 0 for val in x0))

    elif str_ == 'get_payoffs_ambiguity':

        # Distribute input parameters
        simulated, opt = args

        # Check quality of results. As I evaluate the function at the parameters
        # resulting from the optimization, the value of the criterion function
        # should be the same.
        assert (simulated == opt['fun'])

    elif str_ == 'criterion_ambiguity':

        # Distribute input parameters
        simulated, = args

        # Check quality of bounds
        assert (np.isfinite(simulated))

    else:

        raise AssertionError

    # Finishing
    return True
