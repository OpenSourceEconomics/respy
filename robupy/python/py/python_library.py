""" This module contains the PYTHON implementations fo several functions
where FORTRAN alternatives are available.
"""

# standard library
import statsmodels.api as sm
import numpy as np
import logging
import shlex
import os

from scipy.stats import norm

# project library
from robupy.python.py.ambiguity import get_payoffs_ambiguity
from robupy.python.py.auxiliary import get_model_parameters
from robupy.python.py.auxiliary import get_total_value

from robupy.python.py.risk import get_payoffs_risk

from robupy.constants import INTERPOLATION_INADMISSIBLE_STATES
from robupy.constants import MISSING_FLOAT
from robupy.constants import HUGE_FLOAT

from robupy.constants import TINY_FLOAT

import robupy

# Logging
logger = logging.getLogger('ROBUPY_SOLVE')



def likelihood_evaluation(x, edu_max, delta, edu_start, is_debug,
                          is_interpolated, is_python, level, measure, min_idx,
                          num_draws, num_periods, num_points, eps_cholesky,
                          is_ambiguous, seed_solution, shocks, data_array,
                          standard_deviates):
    """ Evaluate likelihood function.
    """
    # Auxiliary objects
    num_agents = int(data_array.shape[0]/num_periods)
    num_sims = standard_deviates.shape[1]

    # Update parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks = \
        get_model_parameters(x, is_debug)

    # Solve the model for updated parametrization
    args = robupy.solve_python(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                shocks, edu_max, delta, edu_start, is_debug, is_interpolated,
                is_python, level, measure, min_idx, num_draws, num_periods,
                num_points, eps_cholesky, is_ambiguous, seed_solution)

    # Distribute return arguments
    mapping_state_idx, periods_emax, periods_future_payoffs = args[:3]
    periods_payoffs_ex_post, periods_payoffs_systematic, states_all = args[3:6]

    # Initialize auxiliary objects
    likl, j = [], 0

    # Calculate the probability over agents and time.
    for i in range(num_agents):
        for period in range(num_periods):
            # Extract observable components of state space as well as agent
            # decision.
            exp_A, exp_B, edu, edu_lagged = data_array[j, 4:].astype(int)
            choice_indicator = data_array[j, 2].astype(int)

            # Transform total years of education to additional years of
            # education and create an index from the choice.
            edu, idx = edu - edu_start, choice_indicator - 1

            # Get state indicator to obtain the systematic component of the
            # agents payoffs. These feed into the simulation of choice
            # probabilities.
            k = mapping_state_idx[period, exp_A, exp_B, edu, edu_lagged]
            payoffs_systematic = periods_payoffs_systematic[period, k, :]

            # Extract relevant deviates from standard normal distribution.
            deviates = standard_deviates[period, :, :].copy()

            # Prepare to calculate product of likelihood contributions.
            likl_contrib = 1.0

            # If an agent is observed working, then the the labor market shocks
            # are observed and the conditional distribution is used to determine
            # the choice probabilities.
            if choice_indicator in [1, 2]:
                # Calculate the disturbance, which follows from a normal
                # distribution.
                eps = np.log(data_array[j, 3].astype(float)) - \
                        np.log(payoffs_systematic[idx])
                # Construct independent normal draws implied by the observed
                # wages.
                if choice_indicator == 1:
                    deviates[:, idx] = eps / np.sqrt(shocks[idx, idx])
                else:
                    deviates[:, idx] = (eps - eps_cholesky[idx, 0] *
                        deviates[:, 0]) / eps_cholesky[idx, idx]
                # Record contribution of wage observation.
                likl_contrib *= norm.pdf(eps, 0.0, np.sqrt(shocks[idx, idx]))

            # Determine conditional deviates. These correspond to the
            # unconditional draws if the agent did not work in the labor market.
            conditional_deviates = np.dot(eps_cholesky, deviates.T).T

            # Simulate the conditional distribution of alternative-specific
            # value functions and determine the choice probabilities.
            counts = np.tile(0.0, 4)

            for s in range(num_sims):
                # Extract deviates from (un-)conditional normal distributions.
                disturbances = conditional_deviates[s, :]

                disturbances[0] = np.exp(disturbances[0])
                disturbances[1] = np.exp(disturbances[1])

                # Calculate total payoff.
                total_payoffs, _, _ = get_total_value(period, num_periods,
                    delta, payoffs_systematic, disturbances, edu_max,
                    edu_start, mapping_state_idx, periods_emax, k, states_all)
                # Record optimal choices
                counts[np.argmax(total_payoffs)] += 1.0

            # Determine relative shares
            choice_probabilities = counts / num_sims

            # Adjust  and record likelihood contribution
            likl_contrib *= choice_probabilities[idx]
            likl += [likl_contrib]

            j += 1

    # Scaling
    likl = -np.mean(np.log(np.clip(likl, TINY_FLOAT, np.inf)))

    # Checks TODO: Refactor
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing
    return likl


def backward_induction(num_periods, max_states_period, periods_eps_relevant,
        num_draws, states_number_period, periods_payoffs_systematic, edu_max,
        edu_start, mapping_state_idx, states_all, delta, is_debug, shocks,
        level, measure, is_interpolated, num_points):
    """ Backward induction procedure. There are two main threads to this
    function depending on whether interpolation is requested or not.
    """
    # Auxiliary objects. These shifts are used to determine the expected
    # values of the two labor market alternatives. These ar log normal
    # distributed and thus the disturbances cannot simply set to zero.
    shifts = [np.exp(shocks[0, 0]/2.0), np.exp(shocks[1, 1]/2.0), 0.0, 0.0]

    # Initialize containers with missing values
    periods_emax = np.tile(MISSING_FLOAT, (num_periods, max_states_period))
    periods_payoffs_ex_post = np.tile(MISSING_FLOAT, (num_periods,
                                               max_states_period, 4))
    periods_future_payoffs = np.tile(MISSING_FLOAT, (num_periods,
                                               max_states_period, 4))

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Extract auxiliary objects
        eps_relevant = periods_eps_relevant[period, :, :]
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
            is_simulated = _get_simulated_indicator(num_points, num_states,
                period, num_periods, is_debug)

            # Constructing the exogenous variable for all states, including the
            # ones where simulation will take place. All information will be
            # used in either the construction of the prediction model or the
            # prediction step.
            exogenous, maxe =_get_exogenous_variables(period, num_periods,
                num_states, delta, periods_payoffs_systematic, shifts,
                edu_max, edu_start, mapping_state_idx, periods_emax, states_all)

            # Constructing the dependent variables for at the random subset of
            # points where the EMAX is actually calculated.
            endogenous = _get_endogenous_variable(period, num_periods,
                num_states, delta, periods_payoffs_systematic, shifts,
                edu_max, edu_start, mapping_state_idx, periods_emax,
                states_all, is_simulated, num_draws, shocks, level, is_debug,
                measure, maxe, eps_relevant)

            # Create prediction model based on the random subset of points where
            # the EMAX is actually simulated and thus dependent and
            # independent variables are available. For the interpolation
            # points, the actual values are used.
            predictions, results = _get_predictions(endogenous, exogenous,
                maxe, is_simulated, num_points, num_states, is_debug)

            # Store results
            periods_emax[period, :num_states] = predictions

        else:

            # Loop over all possible states
            for k in range(states_number_period[period]):

                # Extract payoffs
                payoffs_systematic = periods_payoffs_systematic[period, k, :]

                # Simulate the expected future value.
                emax, payoffs_ex_post, future_payoffs = \
                    get_payoffs(num_draws, eps_relevant, period, k,
                        payoffs_systematic, edu_max, edu_start,
                        mapping_state_idx, states_all, num_periods,
                        periods_emax, delta, is_debug, shocks, level, measure)

                # Store results
                periods_emax[period, k] = emax

                # This information is only available if no interpolation is
                # used. Otherwise all remain set to missing values (see above).
                periods_payoffs_ex_post[period, k, :] = payoffs_ex_post
                periods_future_payoffs[period, k, :] = future_payoffs

    # Finishing. Note that the last two return arguments are not available in
    # for periods, where interpolation is required.
    return periods_emax, periods_payoffs_ex_post, periods_future_payoffs


def get_payoffs(num_draws, eps_relevant, period, k, payoffs_systematic, edu_max,
                edu_start, mapping_state_idx, states_all, num_periods,
                periods_emax, delta, is_debug, shocks, level, measure):
    """ Get payoffs for a particular state.
    """
    # Auxiliary objects
    is_ambiguous = (level > 0.00)

    # Payoffs require different machinery depending on whether there is
    # ambiguity or not.
    if is_ambiguous:
        emax, payoffs_ex_post, future_payoffs = \
                        get_payoffs_ambiguity(num_draws, eps_relevant, period,
                            k, payoffs_systematic, edu_max, edu_start,
                            mapping_state_idx, states_all, num_periods,
                            periods_emax, delta, is_debug, shocks, level,
                            measure)
    else:
        emax, payoffs_ex_post, future_payoffs = \
                        get_payoffs_risk(num_draws, eps_relevant, period, k,
                            payoffs_systematic, edu_max, edu_start,
                            mapping_state_idx, states_all, num_periods,
                            periods_emax, delta, is_debug, shocks, level,
                            measure)

    # Finishing
    return emax, payoffs_ex_post, future_payoffs


def create_state_space(num_periods, edu_start, edu_max, min_idx):
    """ Create grid for state space.
    """
    # Array for possible realization of state space by period
    states_all = np.tile(MISSING_FLOAT, (num_periods, 100000, 4))

    # Array for the mapping of state space values to indices in variety
    # of matrices.
    mapping_state_idx = np.tile(MISSING_FLOAT, (num_periods, num_periods, num_periods,
                                         min_idx, 2))

    # Array for maximum number of realizations of state space by period
    states_number_period = np.tile(MISSING_FLOAT, num_periods)

    # Construct state space by periods
    for period in range(num_periods):

        # Count admissible realizations of state space by period
        k = 0

        # Loop over all admissible work experiences for occupation A
        for exp_A in range(num_periods + 1):

            # Loop over all admissible work experience for occupation B
            for exp_B in range(num_periods + 1):

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
                        total = edu + exp_A + exp_B

                        # Note that the total number of activities does not
                        # have is less or equal to the total possible number of
                        # activities as the rest is implicitly filled with
                        # leisure.
                        if total > period:
                            continue

                        # Collect all possible realizations of state space
                        states_all[period, k, :] = [exp_A, exp_B, edu,
                                                    edu_lagged]

                        # Collect mapping of state space to array index.
                        mapping_state_idx[period, exp_A, exp_B, edu,
                                          edu_lagged] = k

                        # Update count
                        k += 1

        # Record maximum number of state space realizations by time period
        states_number_period[period] = k

    # Finishing
    return states_all, states_number_period, mapping_state_idx


def calculate_payoffs_systematic(num_periods, states_number_period, states_all,
                                edu_start, coeffs_A, coeffs_B, coeffs_edu,
                                coeffs_home, max_states_period):
    """ Calculate ex systematic payoffs.
    """

    # Initialize
    periods_payoffs_systematic = np.tile(MISSING_FLOAT, (num_periods, max_states_period,
                                                  4))

    # Calculate systematic instantaneous payoffs
    for period in range(num_periods - 1, -1, -1):

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Distribute state space
            exp_A, exp_B, edu, edu_lagged = states_all[period, k, :]

            # Auxiliary objects
            covars = [1.0, edu + edu_start, exp_A, exp_A ** 2, exp_B,
                          exp_B ** 2]

            # Calculate systematic part of wages in occupation A
            periods_payoffs_systematic[period, k, 0] = np.exp(
                np.dot(coeffs_A, covars))

            # Calculate systematic part pf wages in occupation B
            periods_payoffs_systematic[period, k, 1] = np.exp(
                np.dot(coeffs_B, covars))

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


def simulate_sample(num_agents, states_all, num_periods,
        mapping_state_idx, periods_payoffs_systematic, periods_eps_relevant,
        edu_max, edu_start, periods_emax, delta):
    """ Sample simulation
    """
    count = 0

    # Initialize data
    dataset = np.tile(MISSING_FLOAT, (num_agents * num_periods, 8))

    for i in range(num_agents):

        current_state = states_all[0, 0, :].copy()

        dataset[count, 0] = i

        # Logging
        if (i != 0) and (i % 100 == 0):
            logger.info('... simulated ' + str(i) + ' agents')

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_A, exp_B, edu, edu_lagged = current_state

            k = mapping_state_idx[period, exp_A, exp_B, edu, edu_lagged]

            # Write agent identifier and current period to data frame
            dataset[count, :2] = i, period

            # Select relevant subset
            payoffs_systematic = periods_payoffs_systematic[period, k, :]
            disturbances = periods_eps_relevant[period, i, :]

            # Get total value of admissible states
            total_payoffs, payoffs_ex_post, _ = get_total_value(period,
                num_periods, delta, payoffs_systematic, disturbances, edu_max,
                edu_start, mapping_state_idx, periods_emax, k, states_all)

            # Determine optimal choice
            max_idx = np.argmax(total_payoffs)

            # Record agent decision
            dataset[count, 2] = max_idx + 1

            # Record earnings
            dataset[count, 3] = MISSING_FLOAT
            if max_idx in [0, 1]:
                dataset[count, 3] = payoffs_ex_post[max_idx]

            # Write relevant state space for period to data frame
            dataset[count, 4:8] = current_state

            # Special treatment for education
            dataset[count, 6] += edu_start

            # Update work experiences and education
            if max_idx == 0:
                current_state[0] += 1
            elif max_idx == 1:
                current_state[1] += 1
            elif max_idx == 2:
                current_state[2] += 1

            # Update lagged education
            current_state[3] = 0

            if max_idx == 2:
                current_state[3] = 1

            # Update row indicator
            count += 1

    # Finishing
    return dataset

''' Private functions
'''


def _logging_prediction_model(results):
    """ Write out some basic information to the solutions log file.
    """
    logger.info('    Information about Prediction Model ')

    string = '''{0:>18}    {1:10.4f} {2:10.4f} {3:10.4f} {4:10.4f}'''
    string += ''' {5:10.4f} {6:10.4f} {7:10.4f} {8:10.4f} {9:10.4f}'''

    logger.info(string.format('Coefficients', *results.params))
    string = '''{0:>18}    {1:10.4f}\n'''

    logger.info(string.format('R-squared', results.rsquared))


def _check_prediction_model(predictions_diff, model, num_points, num_states,
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


def _get_simulated_indicator(num_points, num_candidates, period, num_periods,
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
    is_standardized = is_debug and os.path.isfile('interpolation.txt')
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


def _get_exogenous_variables(period, num_periods, num_states, delta,
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
        expected_values, _, future_payoffs = get_total_value(period, num_periods,
                    delta, payoffs_systematic, shifts, edu_max, edu_start,
                    mapping_state_idx, periods_emax, k, states_all)

        # Treatment of inadmissible states, which will show up in the
        # regression in some way.
        is_inadmissible = (future_payoffs[2] == -HUGE_FLOAT)

        if is_inadmissible:
            expected_values[2] = INTERPOLATION_INADMISSIBLE_STATES

        # Implement level shifts
        maxe[k] = max(expected_values)

        deviations = maxe[k] - expected_values

        exogenous[k, :8] = np.hstack((deviations, np.sqrt(deviations)))

        # Add intercept to set of independent variables and replace
        # infinite values.
        exogenous[:, 8] = 1

    # Finishing
    return exogenous, maxe


def _get_endogenous_variable(period, num_periods, num_states, delta,
        periods_payoffs_systematic, shifts, edu_max, edu_start,
        mapping_state_idx, periods_emax, states_all, is_simulated, num_draws,
        shocks, level, is_debug, measure, maxe, eps_relevant):
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
        emax_simulated, _, _ = get_payoffs(num_draws, eps_relevant, period,
            k, payoffs_systematic, edu_max, edu_start, mapping_state_idx,
            states_all, num_periods, periods_emax, delta, is_debug, shocks,
            level, measure)

        # Construct dependent variable
        endogenous_variable[k] = emax_simulated - maxe[k]

    # Finishing
    return endogenous_variable


def _get_predictions(endogenous, exogenous, maxe, is_simulated, num_points,
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
    _check_prediction_model(endogenous_predicted, model, num_points, num_states,
        is_debug)

    # Write out some basic information to spot problems easily.
    _logging_prediction_model(results)

    # Finishing
    return predictions, results
