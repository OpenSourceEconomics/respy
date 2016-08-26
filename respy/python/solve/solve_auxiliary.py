import os
import shlex

import statsmodels.api as sm
import numpy as np

from respy.python.record.record_solution import record_prediction_model
from respy.python.record.record_solution import record_solution_progress
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.solve.solve_ambiguity import construct_emax_ambiguity
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.solve.solve_risk import construct_emax_risk
from respy.python.shared.shared_constants import MISSING_INT
from respy.python.shared.shared_constants import HUGE_FLOAT


def pyth_create_state_space(num_periods, edu_start, edu_max, min_idx):
    """ Create grid for state space.
    """
    # Array for possible realization of state space by period
    states_all = np.tile(MISSING_INT, (num_periods, 100000, 4))

    # Array for the mapping of state space values to indices in variety
    # of matrices.
    mapping_state_idx = np.tile(MISSING_INT, (num_periods, num_periods,
        num_periods, min_idx, 2))

    # Array for maximum number of realizations of state space by period
    states_number_period = np.tile(MISSING_INT, num_periods)

    # Construct state space by periods
    for period in range(num_periods):

        # Count admissible realizations of state space by period
        k = 0

        # Loop over all admissible work experiences for Occupation A
        for exp_a in range(num_periods + 1):

            # Loop over all admissible work experience for Occupation B
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


def pyth_calculate_rewards_systematic(num_periods, states_number_period,
        states_all, edu_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        max_states_period):
    """ Calculate ex systematic rewards.
    """

    # Initialize
    shape = (num_periods, max_states_period, 4)
    periods_rewards_systematic = np.tile(MISSING_FLOAT, shape)

    # Calculate systematic instantaneous rewards
    for period in range(num_periods - 1, -1, -1):

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Distribute state space
            exp_a, exp_b, edu, edu_lagged = states_all[period, k, :]

            # Auxiliary objects
            covars = [1.0, edu + edu_start, exp_a, exp_a ** 2, exp_b,
                      exp_b ** 2]

            # Calculate systematic part of wages in occupation A
            periods_rewards_systematic[period, k, 0] = \
                np.clip(np.exp(np.dot(coeffs_a, covars)), 0.0, HUGE_FLOAT)

            # Calculate systematic part pf wages in occupation B
            periods_rewards_systematic[period, k, 1] = \
                np.clip(np.exp(np.dot(coeffs_b, covars)), 0.0, HUGE_FLOAT)

            # Calculate systematic part of schooling utility
            reward = coeffs_edu[0]

            # Tuition cost for higher education if agents move
            # beyond high school.
            if edu + edu_start >= 12:
                reward += coeffs_edu[1]

            # Psychic cost of going back to school
            if edu_lagged == 0:
                reward += coeffs_edu[2]

            periods_rewards_systematic[period, k, 2] = reward

            # Calculate systematic part of HOME
            periods_rewards_systematic[period, k, 3] = coeffs_home[0]

    # Finishing
    return periods_rewards_systematic


def pyth_backward_induction(num_periods, max_states_period, periods_draws_emax,
        num_draws_emax, states_number_period, periods_rewards_systematic,
        edu_max, edu_start, mapping_state_idx, states_all, delta, is_debug,
        is_interpolated, num_points_interp, shocks_cholesky, is_ambiguity,
        level):
    """ Backward induction procedure. There are two main threads to this
    function depending on whether interpolation is requested or not.
    """
    # Construct auxiliary objects
    shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)

    # Auxiliary objects. These shifts are used to determine the expected
    # values of the two labor market alternatives. These ar log normal
    # distributed and thus the draws cannot simply set to zero.
    shifts = [0.00, 0.00, 0.00, 0.00]
    shifts[0] = np.clip(np.exp(shocks_cov[0, 0] / 2.0), 0.0, HUGE_FLOAT)
    shifts[1] = np.clip(np.exp(shocks_cov[1, 1] / 2.0), 0.0, HUGE_FLOAT)

    # Initialize containers with missing values
    periods_emax = np.tile(MISSING_FLOAT, (num_periods, max_states_period))

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Extract auxiliary objects
        draws_emax = periods_draws_emax[period, :, :]
        num_states = states_number_period[period]

        draws_emax_transformed = transform_disturbances(draws_emax,
                                    shocks_cholesky)

        record_solution_progress(4, period, num_states)

        # The number of interpolation points is the same for all periods.
        # Thus, for some periods the number of interpolation points is
        # larger than the actual number of states. In that case no
        # interpolation is needed.
        any_interpolated = (num_points_interp <= num_states) and is_interpolated

        # Case distinction
        if any_interpolated:

            # Get indicator for interpolation and simulation of states
            is_simulated = get_simulated_indicator(num_points_interp, num_states,
                period, is_debug)

            # Constructing the exogenous variable for all states, including the
            # ones where simulation will take place. All information will be
            # used in either the construction of the prediction model or the
            # prediction step.
            exogenous, maxe = get_exogenous_variables(period, num_periods,
                num_states, delta, periods_rewards_systematic, shifts,
                edu_max, edu_start, mapping_state_idx, periods_emax, states_all)

            # Constructing the dependent variables for at the random subset of
            # points where the EMAX is actually calculated.
            endogenous = get_endogenous_variable(period, num_periods,
                num_states, delta, periods_rewards_systematic, edu_max,
                edu_start, mapping_state_idx, periods_emax, states_all,
                is_simulated, num_draws_emax, maxe, draws_emax_transformed,
                shocks_cov, level, is_ambiguity)

            # Create prediction model based on the random subset of points where
            # the EMAX is actually simulated and thus dependent and
            # independent variables are available. For the interpolation
            # points, the actual values are used.
            predictions = get_predictions(endogenous, exogenous, maxe,
                is_simulated, num_points_interp, num_states, is_debug)

            # Store results
            periods_emax[period, :num_states] = predictions

        else:

            # Loop over all possible states
            for k in range(states_number_period[period]):

                # Extract rewards
                rewards_systematic = periods_rewards_systematic[period, k, :]

                # Simulate the expected future value.
                if is_ambiguity:
                    emax = construct_emax_ambiguity(num_periods, num_draws_emax,
                        period, k, draws_emax_transformed, rewards_systematic,
                        edu_max, edu_start, periods_emax, states_all,
                        mapping_state_idx, delta, shocks_cov, level)
                else:
                    emax = construct_emax_risk(num_periods, num_draws_emax,
                        period, k, draws_emax_transformed, rewards_systematic,
                        edu_max, edu_start, periods_emax, states_all,
                        mapping_state_idx, delta)

                # Store results
                periods_emax[period, k] = emax

    return periods_emax


def get_simulated_indicator(num_points_interp, num_candidates, period, is_debug):
    """ Get the indicator for points of interpolation and simulation.
    """
    # Drawing random interpolation points
    interpolation_points = np.random.choice(range(num_candidates),
        size=num_points_interp, replace=False)

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
        periods_rewards_systematic, shifts, edu_max, edu_start,
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

        # Extract systematic rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        # Get total value
        total_values = get_total_values(period, num_periods, delta,
            rewards_systematic, shifts, edu_max, edu_start,
            mapping_state_idx, periods_emax, k, states_all)

        # Implement level shifts
        maxe[k] = max(total_values)

        diff = maxe[k] - total_values

        exogenous[k, :8] = np.hstack((diff, np.sqrt(diff)))

        # Add intercept to set of independent variables and replace
        # infinite values.
        exogenous[:, 8] = 1

    # Finishing
    return exogenous, maxe


def get_endogenous_variable(period, num_periods, num_states, delta,
        periods_rewards_systematic, edu_max, edu_start, mapping_state_idx,
        periods_emax, states_all, is_simulated, num_draws_emax, maxe,
        draws_emax_transformed, shocks_cov, level, is_ambiguity):
    """ Construct endogenous variable for the subset of interpolation points.
    """
    # Construct auxiliary objects
    endogenous_variable = np.tile(np.nan, num_states)

    for k in range(num_states):

        # Skip over points that will be interpolated and not simulated.
        if not is_simulated[k]:
            continue

        # Extract rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        # Simulate the expected future value.
        if is_ambiguity:
            emax = construct_emax_ambiguity(num_periods, num_draws_emax,
                period, k, draws_emax_transformed, rewards_systematic,
                edu_max, edu_start, periods_emax, states_all,
                mapping_state_idx, delta, shocks_cov, level)
        else:
            emax = construct_emax_risk(num_periods, num_draws_emax,
                period, k, draws_emax_transformed, rewards_systematic,
                edu_max, edu_start, periods_emax, states_all,
                mapping_state_idx, delta)

        # Construct dependent variable
        endogenous_variable[k] = emax - maxe[k]

    # Finishing
    return endogenous_variable


def get_predictions(endogenous, exogenous, maxe, is_simulated,
        num_points_interp, num_states, is_debug):
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
    check_prediction_model(endogenous_predicted, model, num_points_interp,
        num_states, is_debug)

    # Write out some basic information to spot problems easily.
    record_prediction_model(results)

    # Finishing
    return predictions


def check_prediction_model(predictions_diff, model, num_points_interp, num_states,
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
    if not (is_debug and os.path.exists('interpolation.txt')):
        assert (model.nobs == min(num_points_interp, num_states))


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
        # "_stabilize_myopic" inside "get_total_values".
        assert (delta > 0)

    else:

        raise AssertionError

    # Finishing
    return True


def check_input(respy_obj):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert respy_obj.get_attr('is_locked')

    # Check for previous solution attempt.
    if respy_obj.get_attr('is_solved'):
        respy_obj.reset()

    # Finishing
    return True


