import os
import shlex

import statsmodels.api as sm
import numpy as np

from respy.python.record.record_solution import record_solution_prediction
from respy.python.shared.shared_auxiliary import calculate_rewards_general
from respy.python.shared.shared_auxiliary import calculate_rewards_common
from respy.python.record.record_solution import record_solution_progress
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.solve.solve_risk import construct_emax_risk
from respy.python.shared.shared_constants import HUGE_FLOAT
import pandas as pd


def pyth_create_state_space(num_periods, num_types, edu_spec):
    """ Create state space.

    The state space consists of all admissible combinations of the following elements:

    - period
    - type
    - exp_a
    - exp_b
    - edu
    - choice_lagged

    Parameters
    ----------
    num_periods : int
        Number of periods. ???
    num_types : int
        Number of types. ???
    edu_spec : dict
        Contains educational specification with keys lagged, start, share and max.

    Returns
    -------
    states : pd.DataFrame
        DataFrame containing state space.

    Examples
    --------
    >>> num_periods = 40
    >>> num_types = 1
    >>> edu_spec = {
    ...     "lagged": [1.0], "start": [10], "share": [1.0], "max": 20
    ... }
    >>> df = pyth_create_state_space(num_periods, num_types, edu_spec)
    >>> df.shape
    (324263, 6)
    >>> df.groupby("period").count().iloc[:, 0].values
    array([    2,     4,    19,    47,    92,   158,   249,   369,   522,
             712,   943,  1218,  1535,  1895,  2298,  2744,  3233,  3765,
            4340,  4958,  5619,  6323,  7070,  7860,  8693,  9569, 10488,
           11450, 12455, 13503, 14594, 15728, 16905, 18125, 19388, 20694,
           22043, 23435, 24870, 26348], dtype=int64)

    TODO: Check if range(period + 1) is really necessary.

    """
    # Create list to store state information. Taken from
    # https://stackoverflow.com/a/17496530/7523785.
    data = []

    # Construct state space by periods
    for period in range(num_periods):

        # Loop over all unobserved types
        for type_ in range(num_types):

            # Loop overall all initial levels of schooling
            for edu_start in edu_spec["start"]:

                # Loop over all admissible work experiences for Occupation A
                for exp_a in range(num_periods + 1):

                    # Loop over all admissible work experience for Occupation B
                    for exp_b in range(num_periods + 1):

                        # Loop over all admissible additional education levels
                        for edu_add in range(num_periods + 1):

                            # Check if admissible for time constraints. Note that the
                            # total number of activities does not have is less or equal
                            # to the total possible number of activities as the rest is
                            # implicitly filled with leisure.
                            if edu_add + exp_a + exp_b > period:
                                continue

                            # Agent cannot attain more additional education than
                            # (EDU_MAX - EDU_START).
                            if edu_add > (edu_spec["max"] - edu_start):
                                continue

                            # Loop over all admissible values for the lagged activity:
                            # (1) Occupation A, (2) Occupation B, (3) Education, and (4)
                            # Home.
                            for choice_lagged in [1, 2, 3, 4]:

                                if period > 0:

                                    # (0, 1) Whenever an agent has only worked in
                                    # Occupation A, then the lagged choice cannot be
                                    # anything other than one.
                                    if (choice_lagged != 1) and (
                                        exp_a == period
                                    ):
                                        continue

                                    # (0, 2) Whenever an agent has only worked in
                                    # Occupation B, then the lagged choice cannot be
                                    # anything other than two
                                    if (choice_lagged != 2) and (
                                        exp_b == period
                                    ):
                                        continue

                                    # (0, 3) Whenever an agent has only acquired
                                    # additional education, then the lagged choice
                                    # cannot be anything other than three..
                                    if (choice_lagged != 3) and (
                                        edu_add == period
                                    ):
                                        continue

                                    # (0, 4) Whenever an agent has not acquired any
                                    # additional education and we are not in the first
                                    # period, then lagged activity cannot take a value
                                    # of three.
                                    if (choice_lagged == 3) and (edu_add == 0):
                                        continue

                                # (1, 1) In the first period individual either were in
                                # school the previous period as well or at home. The
                                # cannot have any work experience.
                                if period == 0:
                                    if choice_lagged in [1, 2]:
                                        continue

                                # (2, 1) An individual that has never worked in
                                # Occupation A cannot have that lagged activity.
                                if (choice_lagged == 1) and (exp_a == 0):
                                    continue

                                # (3, 1) An individual that has never worked in
                                # Occupation B cannot have a that lagged activity.
                                if (choice_lagged == 2) and (exp_b == 0):
                                    continue

                                # Store information in a dict and append to data.
                                row = {
                                    "period": period,
                                    "exp_a": exp_a,
                                    "exp_b": exp_b,
                                    "edu": edu_start + edu_add,
                                    "choice_lagged": choice_lagged,
                                    "type": type_,
                                }
                                data.append(row)

    states = pd.DataFrame.from_records(data)

    # If we have multiple initial conditions it might well be the case that we have a
    # duplicate state, i.e. the same state is possible with other initial condition that
    # period.
    states.drop_duplicates(
        subset=["period", "exp_a", "exp_b", "edu", "choice_lagged", "type"],
        keep="first",
        inplace=True,
    )
    states.reset_index(drop=True, inplace=True)

    return states


def pyth_calculate_rewards_systematic(
    states,
    optim_paras,
):
    """ Calculate ex systematic rewards.
    """
    states["intercept"] = 1.0

    # Calculate common and general rewards component.
    states = calculate_rewards_general(
        states, optim_paras
    )
    states = calculate_rewards_common(states, optim_paras)

    # Calculate the systematic part of OCCUPATION A and OCCUPATION B rewards.
    # These are defined in a general sense, where not only wages matter.
    states = calculate_wages_systematic(states, optim_paras)

    states["rewards_a"] = states.wage_a + states.rewards_general_a
    states["rewards_b"] = states.wage_b + states.rewards_general_b

    # Calculate systematic part of SCHOOL rewards
    covariates_education = [
        "intercept", "hs_graduate", "co_graduate", "is_return_not_high_school",
        "is_return_high_school", "period", "is_minor"
    ]

    states["rewards_edu"] = states[covariates_education].dot(optim_paras["coeffs_edu"])

    # Calculate systematic part of HOME
    covariates_home = ["intercept", "is_young_adult", "is_adult"]

    states["rewards_home"] = states[covariates_home].dot(optim_paras["coeffs_home"])

    # Now we add the type-specific deviation for SCHOOL and HOME.
    types_dummy = pd.get_dummies(states.type, prefix="type")
    type_deviations = types_dummy.dot(optim_paras["type_shifts"][:, 2:])

    states.rewards_edu += type_deviations.iloc[:, 0]
    states.rewards_home += type_deviations.iloc[:, 1]

    # We can now also added the common component of rewards.
    states.rewards_a += states.rewards_common
    states.rewards_b += states.rewards_common
    states.rewards_edu += states.rewards_common
    states.rewards_home += states.rewards_common

    states.drop(columns=["intercept"], inplace=True)

    return states


def pyth_backward_induction(
    num_periods,
    is_myopic,
    periods_draws_emax,
    num_draws_emax,
    states,
    is_debug,
    is_interpolated,
    num_points_interp,
    edu_spec,
    optim_paras,
    file_sim,
    is_write,
):
    """ Backward induction procedure. There are two main threads to this function
    depending on whether interpolation is requested or not.
    """
    # Create auxiliary objects for compatibility
    max_states_period = states.loc[states.period.eq(num_periods - 1)].shape[0]
    states_number_period = states.groupby("period").count().iloc[:, 0].values

    # Initialize containers, which contain a lot of missing values as we capture the
    # tree structure in arrays of fixed dimension.
    i, j = num_periods, max_states_period
    periods_emax = np.tile(MISSING_FLOAT, (i, j))

    if is_myopic:
        record_solution_progress(-2, file_sim)

        for period, num_states in enumerate(states_number_period):
            periods_emax[period, :num_states] = 0.0

        return periods_emax

    # Construct auxiliary objects
    shocks_cov = optim_paras["shocks_cholesky"].dot(optim_paras["shocks_cholesky"].T)

    # Auxiliary objects. These shifts are used to determine the expected values of the
    # two labor market alternatives. These are log normal distributed and thus the draws
    # cannot simply set to zero.
    shifts = [0.00, 0.00, 0.00, 0.00]
    shifts[0] = np.clip(np.exp(shocks_cov[0, 0] / 2.0), 0.0, HUGE_FLOAT)
    shifts[1] = np.clip(np.exp(shocks_cov[1, 1] / 2.0), 0.0, HUGE_FLOAT)

    # Initialize containers with missing values
    periods_emax = np.tile(MISSING_FLOAT, (num_periods, max_states_period))

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Extract auxiliary objects
        draws_emax_standard = periods_draws_emax[period, :, :]
        num_states = states_number_period[period]

        # Treatment of the disturbances for the risk-only case is straightforward. Their
        # distribution is fixed once and for all.
        draws_emax_risk = transform_disturbances(
            draws_emax_standard,
            np.tile(0.0, 4),
            optim_paras["shocks_cholesky"],
        )

        if is_write:
            record_solution_progress(4, file_sim, period, num_states)

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In that case no interpolation is needed.
        any_interpolated = (
            num_points_interp <= num_states
        ) and is_interpolated

        # Case distinction
        if any_interpolated:
            # Get indicator for interpolation and simulation of states
            is_simulated = get_simulated_indicator(
                num_points_interp, num_states, period, is_debug
            )

            # Constructing the exogenous variable for all states, including the ones
            # where simulation will take place. All information will be used in either
            # the construction of the prediction model or the prediction step.
            exogenous, maxe = get_exogenous_variables(
                period,
                num_periods,
                num_states,
                periods_rewards_systematic,
                shifts,
                mapping_state_idx,
                periods_emax,
                states_all,
                edu_spec,
                optim_paras,
            )

            # Constructing the dependent variables for at the random subset of points
            # where the EMAX is actually calculated.
            endogenous = get_endogenous_variable(
                period,
                num_periods,
                num_states,
                periods_rewards_systematic,
                mapping_state_idx,
                periods_emax,
                states_all,
                is_simulated,
                num_draws_emax,
                maxe,
                draws_emax_risk,
                edu_spec,
                optim_paras,
            )

            # Create prediction model based on the random subset of points where the
            # EMAX is actually simulated and thus dependent and independent variables
            # are available. For the interpolation points, the actual values are used.
            predictions = get_predictions(
                endogenous, exogenous, maxe, is_simulated, file_sim, is_write
            )

            # Store results
            periods_emax[period, :num_states] = predictions

        else:

            # Loop over all possible states
            for k in range(states_number_period[period]):

                # Extract rewards
                rewards_systematic = periods_rewards_systematic[period, k, :]

                emax = construct_emax_risk(
                    num_periods,
                    num_draws_emax,
                    period,
                    k,
                    draws_emax_risk,
                    rewards_systematic,
                    periods_emax,
                    states_all,
                    mapping_state_idx,
                    edu_spec,
                    optim_paras,
                )

                # Store results
                periods_emax[period, k] = emax

    return periods_emax


def get_simulated_indicator(
    num_points_interp, num_candidates, period, is_debug
):
    """ Get the indicator for points of interpolation and simulation.
    """
    # Drawing random interpolation points
    interpolation_points = np.random.choice(
        range(num_candidates), size=num_points_interp, replace=False
    )

    # Constructing an indicator whether a state will be simulated or interpolated.
    is_simulated = np.tile(False, num_candidates)
    is_simulated[interpolation_points] = True

    # Check for debugging cases.
    is_standardized = is_debug and os.path.exists(".interpolation.respy.test")
    if is_standardized:
        with open(".interpolation.respy.test", "r") as file_:
            indicators = []
            for line in file_:
                indicators += [(shlex.split(line)[period] == "True")]
        is_simulated = indicators[:num_candidates]

    # Type conversion
    is_simulated = np.array(is_simulated)

    # Finishing
    return is_simulated


def get_exogenous_variables(
    period,
    num_periods,
    num_states,
    periods_rewards_systematic,
    shifts,
    mapping_state_idx,
    periods_emax,
    states_all,
    edu_spec,
    optim_paras,
):
    """ Get exogenous variables for interpolation scheme. The unused argument is present
    to align the interface between the PYTHON and FORTRAN implementations.
    """
    # Construct auxiliary objects
    exogenous = np.tile(np.nan, (num_states, 9))
    maxe = np.tile(np.nan, num_states)

    # Iterate over all states.
    for k in range(num_states):

        # Extract systematic rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        # Get total value
        total_values, _ = get_total_values(
            period,
            num_periods,
            optim_paras,
            rewards_systematic,
            shifts,
            edu_spec,
            mapping_state_idx,
            periods_emax,
            k,
            states_all,
        )

        # Implement level shifts
        maxe[k] = max(total_values)

        diff = maxe[k] - total_values

        exogenous[k, :8] = np.hstack((diff, np.sqrt(diff)))

        # Add intercept to set of independent variables and replace infinite values.
        exogenous[:, 8] = 1

    # Finishing
    return exogenous, maxe


def get_endogenous_variable(
    period,
    num_periods,
    num_states,
    periods_rewards_systematic,
    mapping_state_idx,
    periods_emax,
    states_all,
    is_simulated,
    num_draws_emax,
    maxe,
    draws_emax_risk,
    edu_spec,
    optim_paras,
):
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
        emax = construct_emax_risk(
            num_periods,
            num_draws_emax,
            period,
            k,
            draws_emax_risk,
            rewards_systematic,
            periods_emax,
            states_all,
            mapping_state_idx,
            edu_spec,
            optim_paras,
        )

        # Construct dependent variable
        endogenous_variable[k] = emax - maxe[k]

    # Finishing
    return endogenous_variable


def get_predictions(
    endogenous, exogenous, maxe, is_simulated, file_sim, is_write
):
    """ Fit an OLS regression of the exogenous variables on the endogenous variables and use the
    results to predict the endogenous variables for all points in the state space.
    """
    # Define ordinary least squares model and fit to the data.
    model = sm.OLS(endogenous[is_simulated], exogenous[is_simulated])
    results = model.fit()

    # Use the model to predict EMAX for all states. As in Keane & Wolpin (1994), negative
    # predictions are truncated to zero.
    endogenous_predicted = results.predict(exogenous)
    endogenous_predicted = np.clip(endogenous_predicted, 0.00, None)

    # Construct predicted EMAX for all states and the replace interpolation points with simulated
    # values.
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
    assert np.all(predictions_diff >= 0.00)
    assert results.params.shape == (9,)
    assert np.all(np.isfinite(results.params))


def check_input(respy_obj):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert respy_obj.get_attr("is_locked")

    # Check for previous solution attempt.
    if respy_obj.get_attr("is_solved"):
        respy_obj.reset()

    # Finishing
    return True


def calculate_wages_systematic(df, optim_paras):
    """Calculate systematic wages.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing covariates and rewards.
    optim_paras : dict
        ???

    Returns
    -------
    df : pd.DataFrame
        DataFrame with calculated systematic wages.

    TODO: Currently, the function is not variable in occupations.

    """
    df["exp_a_sq"] = df.exp_a ** 2 / 100
    df["exp_b_sq"] = df.exp_b ** 2 / 100

    relevant_covariates = [
        "intercept", "edu", "exp_a", "exp_a_sq", "exp_b", "exp_b_sq",
        "hs_graduate", "co_graduate", "period", "is_minor",
    ]

    # Calculate systematic part of wages in OCCUPATION A and OCCUPATION B
    df["wage_a"] = df[relevant_covariates + ["any_exp_a", "work_a_lagged"]].dot(
        optim_paras["coeffs_a"][:12]
    )
    df.wage_a = np.clip(np.exp(df.wage_a), 0., HUGE_FLOAT)

    df["wage_b"] = df[relevant_covariates + ["any_exp_b", "work_b_lagged"]].dot(
        optim_paras["coeffs_b"][:12]
    )
    df.wage_b = np.clip(np.exp(df.wage_b), 0., HUGE_FLOAT)

    # We need to add the type-specific deviations here as these are part of
    # skill-function component.
    types_dummy = pd.get_dummies(df.type, prefix="type")
    type_deviations = types_dummy.dot(optim_paras["type_shifts"][:, :2])

    df.wage_a = df.wage_a * np.exp(type_deviations.iloc[:, 0])
    df.wage_b = df.wage_b * np.exp(type_deviations.iloc[:, 1])

    return df
