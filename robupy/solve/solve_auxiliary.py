""" This module contains the PYTHON implementations fo several functions
where FORTRAN alternatives are available.
"""

# standard library
import statsmodels.api as sm
import numpy as np

import logging
import shlex
import os

# project library
from robupy.solve.solve_ambiguity import get_payoffs_ambiguity
from robupy.solve.solve_risk import get_payoffs_risk

from robupy.shared.shared_auxiliary import get_total_value

from robupy.shared.shared_constants import INTERPOLATION_INADMISSIBLE_STATES
from robupy.shared.shared_constants import MISSING_FLOAT
from robupy.shared.shared_constants import MISSING_INT
from robupy.shared.shared_constants import HUGE_FLOAT

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')

''' Main functions
'''


def pyth_create_state_space(num_periods, edu_start, edu_max, min_idx):
    """ Create grid for state space.
    """
    # Array for possible realization of state space by period
    states_all = np.tile(MISSING_FLOAT, (num_periods, 100000, 4))

    # Array for the mapping of state space values to indices in variety
    # of matrices.
    mapping_state_idx = np.tile(MISSING_FLOAT, (num_periods, num_periods,
        num_periods, min_idx, 2))

    # Array for maximum number of realizations of state space by period
    states_number_period = np.tile(MISSING_INT, num_periods)

    # Construct state space by periods
    for period in range(num_periods):

        # Count admissible realizations of state space by period
        k = 0

        # Loop over all admissible work experiences for occupation A
        for exp_a in range(num_periods + 1):

            # Loop over all admissible work experience for occupation B
            for exp_b in range(num_periods + 1):

                # Loop over all admissible additional education levels
                for edu in range(num_periods + 1):

                    # Agent cannot attain more additional education
                    # than (EDU_MAX - EDU_START).
                    if edu > (edu_max - edu_start):
                        continue

                    # Loop over all admissible values for leisure. Note that
                    # the leisure variable takes only zero/value. The time path
                    # does not matter.
                    for edu_lagged in [0, 1]:

                        # Check if lagged education admissible. (1) In the
                        # first period all agents have lagged schooling equal
                        # to one.
                        if (edu_lagged == 0) and (period == 0):
                            continue
                        # (2) Whenever an agent has not acquired any additional
                        # education and we are not in the first period,
                        # then this cannot be the case.
                        if (edu_lagged == 1) and (edu == 0) and (period > 0):
                            continue
                        # (3) Whenever an agent has only acquired additional
                        # education, then edu_lagged cannot be zero.
                        if (edu_lagged == 0) and (edu == period):
                            continue

                        # Check if admissible for time constraints
                        total = edu + exp_a + exp_b

                        # Note that the total number of activities does not
                        # have is less or equal to the total possible number of
                        # activities as the rest is implicitly filled with
                        # leisure.
                        if total > period:
                            continue

                        # Collect all possible realizations of state space
                        states_all[period, k, :] = [exp_a, exp_b, edu,
                                                    edu_lagged]

                        # Collect mapping of state space to array index.
                        mapping_state_idx[period, exp_a, exp_b, edu,
                                          edu_lagged] = k

                        # Update count
                        k += 1

        # Record maximum number of state space realizations by time period
        states_number_period[period] = k

    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Collect arguments
    args = (states_all, states_number_period)
    args += (mapping_state_idx, max_states_period,)

    # Finishing
    return args


def pyth_calculate_payoffs_systematic(num_periods, states_number_period,
        states_all, edu_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        max_states_period):
    """ Calculate ex systematic payoffs.
    """

    # Initialize
    shape = (num_periods, max_states_period, 4)
    periods_payoffs_systematic = np.tile(MISSING_FLOAT, shape)

    # Calculate systematic instantaneous payoffs
    for period in range(num_periods - 1, -1, -1):

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Distribute state space
            exp_a, exp_b, edu, edu_lagged = states_all[period, k, :]

            # Auxiliary objects
            covars = [1.0, edu + edu_start, exp_a, exp_a ** 2, exp_b,
                      exp_b ** 2]

            # Calculate systematic part of wages in occupation A
            periods_payoffs_systematic[period, k, 0] = np.exp(
                np.dot(coeffs_a, covars))

            # Calculate systematic part pf wages in occupation B
            periods_payoffs_systematic[period, k, 1] = np.exp(
                np.dot(coeffs_b, covars))

            # Calculate systematic part of schooling utility
            payoff = coeffs_edu[0]

            # Tuition cost for higher education if agents move
            # beyond high school.
            if edu + edu_start >= 12:
                payoff += coeffs_edu[1]

            # Psychic cost of going back to school
            if edu_lagged == 0:
                payoff += coeffs_edu[2]

            periods_payoffs_systematic[period, k, 2] = payoff

            # Calculate systematic part of HOME
            periods_payoffs_systematic[period, k, 3] = coeffs_home[0]

    # Finishing
    return periods_payoffs_systematic


def pyth_backward_induction(num_periods, max_states_period, periods_draws_emax,
        num_draws_emax, states_number_period, periods_payoffs_systematic,
        edu_max, edu_start, mapping_state_idx, states_all, delta, is_debug,
        shocks_cov, level, is_ambiguous, measure, is_interpolated, num_points,
        is_deterministic, shocks_cholesky):
    """ Backward induction procedure. There are two main threads to this
    function depending on whether interpolation is requested or not.
    """
    # Auxiliary objects. These shifts are used to determine the expected
    # values of the two labor market alternatives. These ar log normal
    # distributed and thus the draws cannot simply set to zero.
    shifts = [np.exp(shocks_cov[0, 0] / 2.0), np.exp(shocks_cov[1, 1] / 2.0), 0.0, 0.0]

    # Initialize containers with missing values
    periods_emax = np.tile(MISSING_FLOAT, (num_periods, max_states_period))

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Extract auxiliary objects
        draws_emax = periods_draws_emax[period, :, :]
        num_states = states_number_period[period]

        # Logging.
        string = '''{0[0]:>18}{0[1]:>3}{0[2]:>5}{0[3]:>6}{0[4]:>7}'''
        logger.info(string.format(['... solving period', period, 'with',
            num_states, 'states']))

        # The number of interpolation points is the same for all periods.
        # Thus, for some periods the number of interpolation points is
        # larger than the actual number of states. In that case no
        # interpolation is needed.
        any_interpolated = (num_points <= num_states) and is_interpolated

        # Case distinction
        if any_interpolated:

            # Get indicator for interpolation and simulation of states
            is_simulated = get_simulated_indicator(num_points, num_states,
                period, num_periods, is_debug)

            # Constructing the exogenous variable for all states, including the
            # ones where simulation will take place. All information will be
            # used in either the construction of the prediction model or the
            # prediction step.
            exogenous, maxe = get_exogenous_variables(period, num_periods,
                num_states, delta, periods_payoffs_systematic, shifts,
                edu_max, edu_start, mapping_state_idx, periods_emax, states_all)

            # Constructing the dependent variables for at the random subset of
            # points where the EMAX is actually calculated.
            endogenous = get_endogenous_variable(period, num_periods,
                num_states, delta, periods_payoffs_systematic, edu_max,
                edu_start, mapping_state_idx, periods_emax, states_all,
                is_simulated, num_draws_emax, shocks_cov, level, is_ambiguous,
                is_debug, measure, maxe, draws_emax, is_deterministic,
                shocks_cholesky)

            # Create prediction model based on the random subset of points where
            # the EMAX is actually simulated and thus dependent and
            # independent variables are available. For the interpolation
            # points, the actual values are used.
            predictions, results = get_predictions(endogenous, exogenous,
                maxe, is_simulated, num_points, num_states, is_debug)

            # Store results
            periods_emax[period, :num_states] = predictions

        else:

            # Loop over all possible states
            for k in range(states_number_period[period]):

                # Extract payoffs
                payoffs_systematic = periods_payoffs_systematic[period, k, :]

                # Simulate the expected future value.
                emax = get_payoffs(num_draws_emax, draws_emax, period, k,
                    payoffs_systematic, edu_max, edu_start,
                    mapping_state_idx, states_all, num_periods, periods_emax,
                    delta, is_debug, shocks_cov, level, is_ambiguous, measure,
                    is_deterministic, shocks_cholesky)

                # Store results
                periods_emax[period, k] = emax

    # Finishing. Note that the last two return arguments are not available in
    # for periods, where interpolation is required.
    return periods_emax


''' Auxiliary functions
'''


def get_payoffs(num_draws_emax, draws_emax, period, k, payoffs_systematic,
        edu_max, edu_start, mapping_state_idx, states_all, num_periods,
        periods_emax, delta, is_debug, shocks_cov, level, is_ambiguous,
        measure, is_deterministic, shocks_cholesky):
    """ Get payoffs for a particular state.
    """
    # Payoffs require different machinery depending on whether there is
    # ambiguity or not.
    if is_ambiguous:
        emax = get_payoffs_ambiguity(num_draws_emax, draws_emax,
            period, k, payoffs_systematic, edu_max, edu_start,
            mapping_state_idx, states_all, num_periods, periods_emax,
            delta, is_debug, shocks_cov, level, measure, is_deterministic,
            shocks_cholesky)
    else:
        emax = get_payoffs_risk(num_draws_emax, draws_emax, period, k,
            payoffs_systematic, edu_max, edu_start, mapping_state_idx,
            states_all, num_periods, periods_emax, delta, shocks_cholesky)

    # Finishing
    return emax


def get_simulated_indicator(num_points, num_candidates, period, num_periods,
        is_debug):
    """ Get the indicator for points of interpolation and simulation. The
    unused argument is present to align the interface between the PYTHON and
    FORTRAN implementations.
    """
    # Drawing random interpolation points
    interpolation_points = np.random.choice(range(num_candidates),
        size=num_points, replace=False)

    # Constructing an indicator whether a state will be simulated or
    # interpolated.
    is_simulated = np.tile(False, num_candidates)
    is_simulated[interpolation_points] = True

    # Check for debugging cases.
    is_standardized = is_debug and os.path.exists('interpolation.txt')
    if is_standardized:
        with open('interpolation.txt', 'r') as file_:
            indicators = []
            for line in file_:
                indicators += [(shlex.split(line)[period] == 'True')]
        is_simulated = (indicators[:num_candidates])

    # Type conversion
    is_simulated = np.array(is_simulated)

    # Finishing
    return is_simulated


def get_exogenous_variables(period, num_periods, num_states, delta,
        periods_payoffs_systematic, shifts, edu_max, edu_start,
        mapping_state_idx, periods_emax, states_all):
    """ Get exogenous variables for interpolation scheme. The unused argument
    is present to align the interface between the PYTHON and FORTRAN
    implementations.
    """
    # Construct auxiliary objects
    exogenous = np.tile(np.nan, (num_states, 9))
    maxe = np.tile(np.nan, num_states)

    # Iterate over all states.
    for k in range(num_states):

        # Extract systematic payoff
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        # Get total value
        total_payoffs = get_total_value(period, num_periods, delta,
            payoffs_systematic, shifts, edu_max, edu_start,
            mapping_state_idx, periods_emax, k, states_all)

        # Treatment of inadmissible states, which will show up in the
        # regression in some way.
        is_inadmissible = (total_payoffs[2] == -HUGE_FLOAT)

        if is_inadmissible:
            total_payoffs[2] = INTERPOLATION_INADMISSIBLE_STATES

        # Implement level shifts
        maxe[k] = max(total_payoffs)

        diff = maxe[k] - total_payoffs

        exogenous[k, :8] = np.hstack((diff, np.sqrt(diff)))

        # Add intercept to set of independent variables and replace
        # infinite values.
        exogenous[:, 8] = 1

    # Finishing
    return exogenous, maxe


def get_endogenous_variable(period, num_periods, num_states, delta,
        periods_payoffs_systematic, edu_max, edu_start, mapping_state_idx,
        periods_emax, states_all, is_simulated, num_draws_emax, shocks_cov,
        level, is_ambiguous, is_debug, measure, maxe, draws_emax,
        is_deterministic, shocks_cholesky):
    """ Construct endogenous variable for the subset of interpolation points.
    """
    # Construct auxiliary objects
    endogenous_variable = np.tile(np.nan, num_states)

    for k in range(num_states):

        # Skip over points that will be interpolated and not simulated.
        if not is_simulated[k]:
            continue

        # Extract payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        # Simulate the expected future value.
        emax_simulated = get_payoffs(num_draws_emax, draws_emax, period, k,
            payoffs_systematic, edu_max, edu_start, mapping_state_idx,
            states_all, num_periods, periods_emax, delta, is_debug,
            shocks_cov, level, is_ambiguous, measure, is_deterministic,
            shocks_cholesky)

        # Construct dependent variable
        endogenous_variable[k] = emax_simulated - maxe[k]

    # Finishing
    return endogenous_variable


def get_predictions(endogenous, exogenous, maxe, is_simulated, num_points,
        num_states, is_debug):
    """ Fit an OLS regression of the exogenous variables on the endogenous
    variables and use the results to predict the endogenous variables for all
    points in the state space.
    """
    # Define ordinary least squares model and fit to the data.
    model = sm.OLS(endogenous[is_simulated], exogenous[is_simulated])
    results = model.fit()

    # Use the model to predict EMAX for all states. As in
    # Keane & Wolpin (1994), negative predictions are truncated to zero.
    endogenous_predicted = results.predict(exogenous)
    endogenous_predicted = np.clip(endogenous_predicted, 0.00, None)

    # Construct predicted EMAX for all states and the replace
    # interpolation points with simulated values.
    predictions = endogenous_predicted + maxe
    predictions[is_simulated] = endogenous[is_simulated] + maxe[is_simulated]

    # Checks
    check_prediction_model(endogenous_predicted, model, num_points,
        num_states, is_debug)

    # Write out some basic information to spot problems easily.
    logging_prediction_model(results)

    # Finishing
    return predictions, results

def logging_prediction_model(results):
    """ Write out some basic information to the solutions log file.
    """
    logger.info('    Information about Prediction Model ')

    string = '''{0:>18}    {1:10.4f} {2:10.4f} {3:10.4f} {4:10.4f}'''
    string += ''' {5:10.4f} {6:10.4f} {7:10.4f} {8:10.4f} {9:10.4f}'''

    logger.info(string.format('Coefficients', *results.params))
    string = '''{0:>18}    {1:10.4f}\n'''

    logger.info(string.format('R-squared', results.rsquared))


def logging_solution(which):
    """ Ensure proper handling of logging.
    """
    # Antibugging
    assert (which in ['start', 'stop'])

    # Start logging
    if which == 'start':

        formatter = logging.Formatter('  %(message)s \n')
        logger = logging.getLogger('ROBUPY_SOLVE')
        handler = logging.FileHandler('logging.robupy.sol.log', mode='w',
                                      delay=False)
        handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    elif which == 'stop':
        # Shut down logger and close connection.
        logger = logging.getLogger('ROBUPY_SOLVE')
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    else:
        raise NotImplementedError


def summarize_ambiguity(robupy_obj):
    """ Summarize optimizations in case of ambiguity.
    """

    def _process_cases(list_internal):
        """ Process cases and determine whether keyword or empty line.
        """
        # Antibugging
        assert (isinstance(list_internal, list))

        # Get information
        is_empty_internal = (len(list_internal) == 0)

        if not is_empty_internal:
            is_block_internal = list_internal[0].isupper()
        else:
            is_block_internal = False

        # Antibugging
        assert (is_block_internal in [True, False])
        assert (is_empty_internal in [True, False])

        # Finishing
        return is_empty_internal, is_block_internal

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    dict_ = dict()

    for line in open('ambiguity.robupy.log').readlines():

        # Split line
        list_ = shlex.split(line)

        # Determine special cases
        is_empty, is_block = _process_cases(list_)

        # Applicability
        if is_empty:
            continue

        # Prepare dictionary
        if is_block:

            period = int(list_[1])

            if period in dict_.keys():
                continue

            dict_[period] = {}
            dict_[period]['success'] = 0
            dict_[period]['failure'] = 0
            dict_[period]['total'] = 0

        # Collect success indicator
        if list_[0] == 'Success':
            dict_[period]['total'] += 1

            is_success = (list_[1] == 'True')
            if is_success:
                dict_[period]['success'] += 1
            else:
                dict_[period]['failure'] += 1

    with open('ambiguity.robupy.log', 'a') as file_:

        file_.write('\nSUMMARY\n\n')

        string = '''{0[0]:>10} {0[1]:>10} {0[2]:>10} {0[3]:>10}\n'''
        file_.write(string.format(['Period', 'Total', 'Success', 'Failure']))

        file_.write('\n')

        for period in range(num_periods):
            total = dict_[period]['total']

            success = dict_[period]['success']/total
            failure = dict_[period]['failure']/total

            string = '''{0[0]:>10} {0[1]:>10} {0[2]:10.2f} {0[3]:10.2f}\n'''
            file_.write(string.format([period, total, success, failure]))


def cleanup():
    """ Cleanup all selected files. Note that not simply all *.robupy.*
    files can be deleted as the blank logging files are already created.
    """
    if os.path.exists('ambiguity.robupy.log'):
        os.unlink('ambiguity.robupy.log')


def start_ambiguity_logging(is_ambiguous, is_debug):
    """ Start logging for ambiguity.
    """
    # Start logging if required
    if os.path.exists('ambiguity.robupy.log'):
        os.remove('ambiguity.robupy.log')

    if is_debug and is_ambiguous:
        open('ambiguity.robupy.log', 'w').close()


def check_prediction_model(predictions_diff, model, num_points, num_states,
        is_debug):
    """ Perform some basic consistency checks for the prediction model.
    """
    # Construct auxiliary object
    results = model.fit()
    # Perform basic checks
    assert (np.all(predictions_diff >= 0.00))
    assert (results.params.shape == (9,))
    assert (np.all(np.isfinite(results.params)))

    # Check for standardization as the following constraint is not
    # necessarily satisfied in that case. For ease of application, we do not
    # ensure that the same number of interpolation points is available.
    if not (is_debug and os.path.isfile('interpolation.txt')):
        assert (model.nobs == min(num_points, num_states))


def checks(str_, *args):
    """ Some guards to the interfaces.
    """
    if str_ == '_backward_induction_procedure':

        # Distribute input parameters
        delta, = args

        # The backward induction procedure does not work properly for the
        # myopic case anymore. This is necessary as in the special
        # case where delta is equal to zero, (-np.inf * 0.00) evaluates to
        # NAN. This is returned as the maximum value when calling np.argmax.
        # This was preciously handled by an auxiliary function
        # "_stabilize_myopic" inside "get_total_value".
        assert (delta > 0)

    else:

        raise AssertionError

    # Finishing
    return True


def check_input(robupy_obj):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert robupy_obj.get_attr('is_locked')

    # Check for previous solution attempt.
    assert (not robupy_obj.get_attr('is_solved'))

    # Finishing
    return True
