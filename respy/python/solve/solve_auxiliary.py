import os
import shlex

import statsmodels.api as sm
import numpy as np

from respy.python.record.record_solution import record_solution_prediction
from respy.python.shared.shared_auxiliary import calculate_rewards_general
from respy.python.shared.shared_auxiliary import calculate_rewards_common
from respy.python.record.record_solution import record_solution_progress
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.solve.solve_ambiguity import construct_emax_ambiguity
from respy.python.shared.shared_auxiliary import construct_covariates
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_constants import MIN_AMBIGUITY
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.solve.solve_risk import construct_emax_risk
from respy.python.shared.shared_constants import MISSING_INT
from respy.python.shared.shared_constants import HUGE_FLOAT


def pyth_create_state_space(num_periods, num_types, edu_spec):
    """ Create grid for state space.
    """
    # Auxiliary information
    min_idx = edu_spec['max'] + 1

    # Array for possible realization of state space by period
    states_all = np.tile(MISSING_INT, (num_periods, 100000, 5))

    # Array for the mapping of state space values to indices in variety of matrices.
    shape = (num_periods, num_periods, num_periods, min_idx, 4, num_types)
    mapping_state_idx = np.tile(MISSING_INT, shape)

    # Array for maximum number of realizations of state space by period
    states_number_period = np.tile(MISSING_INT, num_periods)

    # Construct state space by periods
    for period in range(num_periods):

        # Count admissible realizations of state space by period
        k = 0

        # Loop over all unobserved types
        for type_ in range(num_types):

            # Loop overall all initial levels of schooling
            for edu_start in edu_spec['start']:

                # Loop over all admissible work experiences for Occupation A
                for exp_a in range(num_periods + 1):

                    # Loop over all admissible work experience for Occupation B
                    for exp_b in range(num_periods + 1):

                        # Loop over all admissible additional education levels
                        for edu_add in range(num_periods + 1):

                            # Check if admissible for time constraints. Note that the total
                            # number of activities does not have is less or equal to the
                            # total possible number of activities as the rest is implicitly
                            # filled with leisure.
                            if edu_add + exp_a + exp_b > period:
                                continue

                            # Agent cannot attain more additional education than (EDU_MAX -
                            # EDU_START).
                            if edu_add > (edu_spec['max'] - edu_start):
                                continue

                            # Loop over all admissible values for the lagged activity: (0) Home,
                            # (1) Education, (2) Occupation A, and (3) Occupation B.
                            for activity_lagged in [0, 1, 2, 3]:

                                # (0, 1) Whenever an agent has only acquired additional education,
                                # then activity_lagged cannot be zero.
                                if (activity_lagged == 0) and (edu_add == period):
                                    continue

                                # (0, 2) Whenever an agent has only worked in Occupation A,
                                # then activity_lagged cannot be zero.
                                if (activity_lagged == 0) and (exp_a == period):
                                    continue

                                # (0, 3) Whenever an agent has only worked in Occupation B,
                                # then activity lagged cannot be zero.
                                if (activity_lagged == 0) and (exp_b == period):
                                    continue

                                # (1, 1) In the first period all agents have lagged schooling equal
                                # to one and nothing else.
                                if (activity_lagged != 1) and (period == 0):
                                    continue

                                # (1, 2) Whenever an agent has not acquired any additional education
                                # and we are not in the first period, then lagged activity
                                # cannot take a value of one.
                                if (activity_lagged == 1) and (edu_add == 0) and (period > 0):
                                    continue

                                # (2, 1) An individual that has never worked in Occupation A
                                # cannot have a that lagged activity.
                                if (activity_lagged == 2) and (exp_a == 0):
                                    continue

                                # (3, 1) An individual that has never worked in Occupation B
                                # cannot have a that lagged activity.
                                if (activity_lagged == 3) and (exp_b == 0):
                                    continue

                                # If we have multiple initial conditions it might well be the
                                # case that we have a duplicate state, i.e. the same state is
                                # possible with other initial condition that period.
                                if mapping_state_idx[period, exp_a, exp_b, edu_start + edu_add,
                                                     activity_lagged, type_] != MISSING_INT:
                                    continue

                                # Collect mapping of state space to array index.
                                mapping_state_idx[period, exp_a, exp_b, edu_start + edu_add,
                                                  activity_lagged, type_] = k

                                # Collect all possible realizations of state space
                                states_all[period, k, :] = [exp_a, exp_b, edu_start + edu_add,
                                                            activity_lagged, type_]

                                # Update count
                                k += 1

        # Record maximum number of state space realizations by time period
        states_number_period[period] = k

    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Collect arguments
    args = (states_all, states_number_period, mapping_state_idx, max_states_period,)

    # Finishing
    return args


def pyth_calculate_rewards_systematic(num_periods, states_number_period, states_all,
        max_states_period, optim_paras):
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
            exp_a, exp_b, edu, activity_lagged, type_ = states_all[period, k, :]

            # Initialize container
            rewards = np.tile(np.nan, 4)

            # Construct auxiliary information
            covariates = construct_covariates(exp_a, exp_b, edu, activity_lagged, type_, period)

            # Calculate common and general rewards component.
            rewards_general = calculate_rewards_general(covariates, optim_paras)
            rewards_common = calculate_rewards_common(covariates, optim_paras)

            # Calculate the systematic part of OCCUPATION A and OCCUPATION B rewards. These are
            # defined in a general sense, where not only wages matter.
            wages = calculate_wages_systematic(covariates, optim_paras)

            for j in [0, 1]:
                rewards[j] = wages[j] + rewards_general[j]

            # Calculate systematic part of SCHOOL rewards
            covars_edu = []
            covars_edu += [1]
            covars_edu += [covariates['is_return_not_high_school']]
            covars_edu += [covariates['is_return_high_school']]
            covars_edu += [covariates['period']]
            covars_edu += [covariates['is_minor']]

            rewards[2] = np.dot(optim_paras['coeffs_edu'], covars_edu)

            # Calculate systematic part of HOME
            covars_home = []
            covars_home += [1]
            covars_home += [covariates['is_young_adult']]   # Age 18 - 20
            covars_home += [covariates['is_adult']]         # Age 21 and over

            rewards[3] = np.dot(optim_paras['coeffs_home'], covars_home)

            # Now we add the type-specific deviation for SCHOOL and HOME.
            for j in [2, 3]:
                rewards[j] = rewards[j] + optim_paras['type_shifts'][type_, j]

            # We can now also added the common component of rewards.
            for j in range(4):
                rewards[j] = rewards[j] + rewards_common

            periods_rewards_systematic[period, k, :] = rewards

    # Finishing
    return periods_rewards_systematic


def pyth_backward_induction(num_periods, is_myopic, max_states_period, periods_draws_emax,
        num_draws_emax, states_number_period, periods_rewards_systematic, mapping_state_idx,
        states_all, is_debug, is_interpolated, num_points_interp, edu_spec, ambi_spec, optim_paras,
        optimizer_options, file_sim, is_write):
    """ Backward induction procedure. There are two main threads to this function depending on 
    whether interpolation is requested or not.
    """
    # Initialize containers, which contain a lot of missing values as we capture the tree
    # structure in arrays of fixed dimension.
    i, j = num_periods, max_states_period
    opt_ambi_details = np.tile(MISSING_FLOAT, (i, j, 7))
    periods_emax = np.tile(MISSING_FLOAT, (i, j))

    if is_myopic:
        record_solution_progress(-2, file_sim)

        for period, num_states in enumerate(states_number_period):
            periods_emax[period, :num_states] = 0.0

        return periods_emax, opt_ambi_details

    # Construct auxiliary objects
    shocks_cov = np.matmul(optim_paras['shocks_cholesky'], optim_paras['shocks_cholesky'].T)

    # Auxiliary objects. These shifts are used to determine the expected values of the two labor
    # market alternatives. These are log normal distributed and thus the draws cannot simply set
    # to zero.
    shifts = [0.00, 0.00, 0.00, 0.00]
    shifts[0] = np.clip(np.exp(shocks_cov[0, 0] / 2.0), 0.0, HUGE_FLOAT)
    shifts[1] = np.clip(np.exp(shocks_cov[1, 1] / 2.0), 0.0, HUGE_FLOAT)

    # Initialize containers with missing values
    periods_emax = np.tile(MISSING_FLOAT, (num_periods, max_states_period))

    draws_emax_ambiguity_transformed = np.tile(MISSING_FLOAT, (num_draws_emax, 4))
    draws_emax_ambiguity_standard = np.tile(MISSING_FLOAT, (num_draws_emax, 4))
    draws_emax_risk = np.tile(MISSING_FLOAT, (num_draws_emax, 4))

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Extract auxiliary objects
        draws_emax_standard = periods_draws_emax[period, :, :]
        num_states = states_number_period[period]

        # We prepare two sets of disturbances for the case of ambiguity or risk, depending on
        # what the relevant request requires.
        if optim_paras['level'] > MIN_AMBIGUITY:
            if ambi_spec['mean']:
                draws_emax_ambiguity_transformed = \
                        np.dot(optim_paras['shocks_cholesky'], draws_emax_standard.T).T
            else:
                draws_emax_ambiguity_standard = draws_emax_standard.copy()
        else:
            # Treatment of the disturbances for the risk-only case is straightforward. Their
            # distribution is fixed once and for all.
            draws_emax_risk = transform_disturbances(draws_emax_standard,
                np.array([0.0, 0.0, 0.0, 0.0]), optim_paras['shocks_cholesky'])

        if is_write:
            record_solution_progress(4, file_sim, period, num_states)

        # The number of interpolation points is the same for all periods. Thus, for some periods
        # the number of interpolation points is larger than the actual number of states. In that
        # case no interpolation is needed.
        any_interpolated = (num_points_interp <= num_states) and is_interpolated

        # Case distinction
        if any_interpolated:
            # Get indicator for interpolation and simulation of states
            is_simulated = get_simulated_indicator(num_points_interp, num_states, period, is_debug)

            # Constructing the exogenous variable for all states, including the ones where
            # simulation will take place. All information will be used in either the construction
            #  of the prediction model or the prediction step.
            exogenous, maxe = get_exogenous_variables(period, num_periods, num_states,
                periods_rewards_systematic, shifts, mapping_state_idx, periods_emax, states_all,
                edu_spec, optim_paras)

            # Constructing the dependent variables for at the random subset of points where the
            # EMAX is actually calculated.
            endogenous, opt_ambi_details = get_endogenous_variable(period, num_periods, num_states,
                periods_rewards_systematic, mapping_state_idx, periods_emax, states_all,
                is_simulated, num_draws_emax, maxe, draws_emax_risk, draws_emax_ambiguity_standard,
                draws_emax_ambiguity_transformed, edu_spec, ambi_spec, optim_paras,
                optimizer_options, opt_ambi_details)

            # Create prediction model based on the random subset of points where the EMAX is
            # actually simulated and thus dependent and independent variables are available. For
            # the interpolation points, the actual values are used.
            predictions = get_predictions(endogenous, exogenous, maxe,
                is_simulated, file_sim, is_write)

            # Store results
            periods_emax[period, :num_states] = predictions

        else:

            # Loop over all possible states
            for k in range(states_number_period[period]):

                # Extract rewards
                rewards_systematic = periods_rewards_systematic[period, k, :]

                # Simulate the expected future value.
                if optim_paras['level'] > MIN_AMBIGUITY:
                    emax, opt_ambi_details = construct_emax_ambiguity(num_periods, num_draws_emax,
                        period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed,
                        rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec,
                        ambi_spec, optim_paras, optimizer_options, opt_ambi_details)
                else:
                    emax = construct_emax_risk(num_periods, num_draws_emax, period, k,
                        draws_emax_risk, rewards_systematic, periods_emax, states_all,
                        mapping_state_idx, edu_spec, optim_paras)

                # Store results
                periods_emax[period, k] = emax

    return periods_emax, opt_ambi_details


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
    is_standardized = is_debug and os.path.exists('.interpolation.respy.test')
    if is_standardized:
        with open('.interpolation.respy.test', 'r') as file_:
            indicators = []
            for line in file_:
                indicators += [(shlex.split(line)[period] == 'True')]
        is_simulated = (indicators[:num_candidates])

    # Type conversion
    is_simulated = np.array(is_simulated)

    # Finishing
    return is_simulated


def get_exogenous_variables(period, num_periods, num_states, periods_rewards_systematic, shifts,
        mapping_state_idx, periods_emax, states_all, edu_spec, optim_paras):
    """ Get exogenous variables for interpolation scheme. The unused argument is present to align 
    the interface between the PYTHON and FORTRAN implementations.
    """
    # Construct auxiliary objects
    exogenous = np.tile(np.nan, (num_states, 9))
    maxe = np.tile(np.nan, num_states)

    # Iterate over all states.
    for k in range(num_states):

        # Extract systematic rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        # Get total value
        total_values = get_total_values(period, num_periods, optim_paras, rewards_systematic,
            shifts, edu_spec, mapping_state_idx, periods_emax, k, states_all)

        # Implement level shifts
        maxe[k] = max(total_values)

        diff = maxe[k] - total_values

        exogenous[k, :8] = np.hstack((diff, np.sqrt(diff)))

        # Add intercept to set of independent variables and replace infinite values.
        exogenous[:, 8] = 1

    # Finishing
    return exogenous, maxe


def get_endogenous_variable(period, num_periods, num_states, periods_rewards_systematic,
        mapping_state_idx, periods_emax, states_all, is_simulated, num_draws_emax, maxe,
        draws_emax_risk, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, edu_spec,
        ambi_spec, optim_paras, optimizer_options, opt_ambi_details):
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
        if optim_paras['level'] > MIN_AMBIGUITY:
            emax, opt_ambi_details = construct_emax_ambiguity(num_periods, num_draws_emax, period,
                k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed,
                rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec,
                ambi_spec, optim_paras, optimizer_options, opt_ambi_details)
        else:
            emax = construct_emax_risk(num_periods, num_draws_emax, period, k, draws_emax_risk,
                rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec,
                optim_paras)

        # Construct dependent variable
        endogenous_variable[k] = emax - maxe[k]

    # Finishing
    return endogenous_variable, opt_ambi_details


def get_predictions(endogenous, exogenous, maxe, is_simulated, file_sim,
        is_write):
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
    check_prediction_model(endogenous_predicted, model)

    # Write out some basic information to spot problems easily.
    if is_write:
        record_solution_prediction(results, file_sim)

    # Finishing
    return predictions


def check_prediction_model(predictions_diff, model):
    """ Perform some basic consistency checks for the prediction model.
    """
    # Construct auxiliary object
    results = model.fit()
    # Perform basic checks
    assert (np.all(predictions_diff >= 0.00))
    assert (results.params.shape == (9,))
    assert (np.all(np.isfinite(results.params)))


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


def calculate_wages_systematic(covariates, optim_paras):
    """ Calculate the systematic component of wages.
    """
    # Collect all relevant covariates
    covars_wages = []
    covars_wages += [1.0]
    covars_wages += [covariates['edu']]
    covars_wages += [covariates['exp_a']]
    covars_wages += [(covariates['exp_a'] ** 2) / 100.00]
    covars_wages += [covariates['exp_b']]
    covars_wages += [(covariates['exp_b'] ** 2) / 100.00]
    covars_wages += [covariates['hs_graduate']]
    covars_wages += [covariates['co_graduate']]
    covars_wages += [covariates['period']]
    covars_wages += [covariates['is_minor']]

    # This used for testing purposes, where we compare the results from the RESPY package
    # to the original RESTUD program.
    if os.path.exists('.restud.respy.scratch'):
        covars_wages[3] *= 100.00
        covars_wages[5] *= 100.00

    # Calculate systematic part of wages in OCCUPATION A and OCCUPATION B
    wages = np.tile(np.nan, 2)

    wage = np.exp(np.dot(optim_paras['coeffs_a'][:10], covars_wages))
    wages[0] = np.clip(wage, 0.0, HUGE_FLOAT)

    wage = np.exp(np.dot(optim_paras['coeffs_b'][:10], covars_wages))
    wages[1] = np.clip(wage, 0.0, HUGE_FLOAT)

    # We need to add the type-specific deviations here as these are part of
    # skill-function component.
    for j in [0, 1]:
        wages[j] = wages[j] * np.exp(optim_paras['type_shifts'][covariates['type'], j])

    return wages
