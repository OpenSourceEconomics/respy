""" This module contains the functions related to the incorporation of
    ambiguity in the model.
"""

# standard library
from scipy.optimize import minimize
import numpy as np

# project library
from robupy.python.py.auxiliary import simulate_emax

from robupy.constants import HUGE_FLOAT

''' Public functions
'''


def get_payoffs_ambiguity(num_draws, eps_relevant, period, k, payoffs_systematic,
        edu_max, edu_start, mapping_state_idx, states_all, num_periods,
        periods_emax, delta, is_debug, shocks, level, measure):
    """ Get worst case
    """
    opt = _determine_worst_case(num_draws, eps_relevant, period, k, payoffs_systematic,
        edu_max, edu_start, mapping_state_idx, states_all, num_periods,
        periods_emax, delta, is_debug, shocks, level, measure)

    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant_emax = transform_disturbances_ambiguity(eps_relevant, opt['x'])

    simulated, payoffs_ex_post, future_payoffs = \
        simulate_emax(num_periods, num_draws, period, k, eps_relevant_emax,
            payoffs_systematic, edu_max, edu_start, periods_emax, states_all,
            mapping_state_idx, delta)

    # Debugging. This only works in the case of success, as otherwise
    # opt['fun'] is not equivalent to simulated.
    if is_debug and opt['success']:
        checks_ambiguity('get_payoffs_ambiguity', simulated, opt)

    # Finishing
    return simulated, payoffs_ex_post, future_payoffs

''' Private functions
'''


def _determine_worst_case(num_draws, eps_relevant, period, k, payoffs_systematic,
        edu_max, edu_start, mapping_state_idx, states_all, num_periods,
        periods_emax, delta, is_debug, shocks, level, measure):
    """ Determine the worst case outcome for the given parameterization.
    """
    # Initialize options.
    options = dict()
    options['maxiter'] = 100000000

    # Initialize optimization problem.
    x0 = _get_start(is_debug)

    # Collect arguments
    args = (num_draws, eps_relevant, period, k, payoffs_systematic, edu_max,
            edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
            delta)

    # Run optimization
    if measure == 'absolute':

        bounds = _prep_absolute(level, is_debug)

        opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                       bounds=bounds)

    else:

        constraints = _prep_kl(shocks, level)

        opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                       constraints=constraints)

        # This is not very satisfactory, but it occurs that the constraint
        # is not satisfied, even though success is indicated. To ensure
        # a smooth and informative run of TEST_F in the random development
        # test battery the following checks are performed.
        if is_debug:
            opt = _correct_debugging(opt, x0, shocks, level, eps_relevant,
                        num_periods, num_draws, period, k, payoffs_systematic,
                        edu_max, edu_start, periods_emax, states_all,
                        mapping_state_idx, delta)

        # Stabilization. If the optimization fails the starting values are
        # used otherwise it happens that the constraint is not satisfied by far.
        if not opt['success']:
            opt['x'] = x0

    # Logging result to file
    if is_debug:
        # Evaluate divergence at final value.
        div = _divergence(opt['x'], shocks, level) - level
        _write_result(period, k, opt, div)

    # Finishing
    return opt


def _correct_debugging(opt, x0, shocks, level, eps_relevant, num_periods,
        num_draws, period, k, payoffs_systematic, edu_max, edu_start, periods_emax,
        states_all, mapping_state_idx, delta):
    """ Some manipulations for test battery
    """
    # Check applicability
    if not (level < 0.1e-10):
        return opt

    # Distribute results
    is_success = opt['success']
    x = opt['x']

    # Check if divergence actually not satisfied while success is indicated.
    if is_success and (_divergence(x, shocks, level) < 0):

        # Correct resulting values
        opt['x'] = x0

        # Correct final function value
        eps_relevant_emax = transform_disturbances_ambiguity(eps_relevant, opt['x'])

        simulated, payoffs_ex_post, future_payoffs = \
                    simulate_emax(num_periods, num_draws, period, k,
                        eps_relevant_emax, payoffs_systematic, edu_max, edu_start,
                        periods_emax, states_all, mapping_state_idx, delta)

        opt['fun'] = simulated

    # Finishing
    return opt


def transform_disturbances_ambiguity(eps_relevant, x):
    """ Transform disturbances
    """
    # Initialize clean slate
    eps_relevant_emax = eps_relevant.copy()

    # Mean shift due to ambiguity
    eps_relevant_emax[:, :2] = eps_relevant_emax[:, :2] + x

    # Exponentiation for occupations
    for j in [0, 1]:
        eps_relevant_emax[:, j] = np.clip(np.exp(eps_relevant_emax[:, j]),
                                          0.0, HUGE_FLOAT)

    # Finishing
    return eps_relevant_emax


def _prep_kl(shocks, level):
    """ Construct Kullback-Leibler constraint for optimization.
    """
    # Construct constraint
    constraint_divergence = dict()

    constraint_divergence['type'] = 'eq'

    constraint_divergence['fun'] = _divergence

    constraint_divergence['args'] = (shocks, level)

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


def _criterion(x, num_draws, eps_relevant, period, k, payoffs_systematic, edu_max,
        edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
        delta):
    """ Simulate expected future value for alternative shock distributions.
    """

    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant_emax = transform_disturbances_ambiguity(eps_relevant, x)

    # Simulate the expected future value for a given parametrization.
    simulated, _, _ = simulate_emax(num_periods, num_draws, period, k,
                        eps_relevant_emax, payoffs_systematic, edu_max, edu_start,
                        periods_emax, states_all, mapping_state_idx, delta)
    # Debugging
    checks_ambiguity('_criterion', simulated)

    # Finishing
    return simulated


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
        checks_ambiguity('_get_start', x0)

    # Finishing
    return x0


def _prep_absolute(level, is_debug):
    """ Get bounds.
    """
    # Construct appropriate bounds
    bounds = [[-level, level], [-level, level]]

    # Debugging
    if is_debug:
        checks_ambiguity('_prep_absolute', bounds)

    # Finishing
    return bounds


def checks_ambiguity(str_, *args):
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

    elif str_ == '_criterion':

        # Distribute input parameters
        simulated, = args

        # Check quality of bounds
        assert (np.isfinite(simulated))

    elif str_ == '_prep_absolute':

        # Distribute input parameters
        bounds, = args

        # Check quality of bounds
        assert (len(bounds) == 2)

        for i in range(2):
            assert (bounds[0] == bounds[i])

    else:

        raise AssertionError

    # Finishing
    return True
