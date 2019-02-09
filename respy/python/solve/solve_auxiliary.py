import os
import shlex

import statsmodels.api as sm
import numpy as np

from respy.python.record.record_solution import record_solution_prediction
from respy.python.shared.shared_auxiliary import calculate_rewards_general
from respy.python.shared.shared_auxiliary import calculate_rewards_common
from respy.python.record.record_solution import record_solution_progress
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.shared.shared_auxiliary import get_emaxs_of_subsequent_period
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

                # For occupations and education it is necessary to loop over period
                # + 1 as zero has to be included if it is never this choice and period
                # + 1 if it is always the same choice.

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
                                    # cannot be anything other than three.
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
                                # school the previous period as well or at home. They
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

                                # Store information in a dictionary and append to data.
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


def pyth_calculate_rewards_systematic(states, optim_paras):
    """ Calculate ex systematic rewards.
    """
    states["intercept"] = 1.0

    # Calculate common and general rewards component.
    states = calculate_rewards_general(states, optim_paras)
    states = calculate_rewards_common(states, optim_paras)

    # Calculate the systematic part of OCCUPATION A and OCCUPATION B rewards.
    # These are defined in a general sense, where not only wages matter.
    states = calculate_wages_systematic(states, optim_paras)

    states["rewards_a"] = states.wage_a + states.rewards_general_a
    states["rewards_b"] = states.wage_b + states.rewards_general_b

    # Calculate systematic part of SCHOOL rewards
    covariates_education = [
        "intercept",
        "hs_graduate",
        "co_graduate",
        "is_return_not_high_school",
        "is_return_high_school",
        "period",
        "is_minor",
    ]

    states["rewards_edu"] = states[covariates_education].dot(
        optim_paras["coeffs_edu"]
    )

    # Calculate systematic part of HOME
    covariates_home = ["intercept", "is_young_adult", "is_adult"]

    states["rewards_home"] = states[covariates_home].dot(
        optim_paras["coeffs_home"]
    )

    # Now we add the type-specific deviation for SCHOOL and HOME.
    types_dummy = pd.get_dummies(states.type, prefix="type")
    type_deviations = types_dummy.dot(optim_paras["type_shifts"][:, 2:])

    states.rewards_edu += type_deviations.iloc[:, 0]
    states.rewards_home += type_deviations.iloc[:, 1]

    # We can now also added the common component of rewards.
    states["rewards_systematic_a"] = states.rewards_a + states.rewards_common
    states["rewards_systematic_b"] = states.rewards_b + states.rewards_common
    states["rewards_systematic_edu"] = (
        states.rewards_edu + states.rewards_common
    )
    states["rewards_systematic_home"] = (
        states.rewards_home + states.rewards_common
    )

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
    """ Calculate utilities with backward induction.

    Parameters
    ----------
    num_periods : int
    is_myopic : bool
    periods_draws_emax : ???
    num_draws_emax : int
    states : pd.DataFrame
    is_debug : bool
    is_interpolated : np.array
    num_points_interp : int
    edu_spec : dict
    optim_paras : dict
    file_sim : ???
    is_write : ???

    Returns
    -------
    states : pd.DataFrame

    """
    # Create auxiliary objects for compatibility
    states_number_period = states.groupby("period").count().iloc[:, 0].values

    if is_myopic:
        record_solution_progress(-2, file_sim)

        states["emax"] = 0.0

        return states

    # Construct auxiliary objects
    shocks_cov = optim_paras["shocks_cholesky"].dot(
        optim_paras["shocks_cholesky"].T
    )

    # Auxiliary objects. These shifts are used to determine the expected values of the
    # two labor market alternatives. These are log normal distributed and thus the draws
    # cannot simply set to zero.
    shifts = [0.00, 0.00, 0.00, 0.00]
    shifts[0] = np.clip(np.exp(shocks_cov[0, 0] / 2.0), 0.0, HUGE_FLOAT)
    shifts[1] = np.clip(np.exp(shocks_cov[1, 1] / 2.0), 0.0, HUGE_FLOAT)

    for period in reversed(range(num_periods)):

        # Get future utility values. Set them to zero for the last period.
        if period == states.period.max():
            for col in ["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"]:
                states.loc[states.period.eq(period), col] = 0.0
        else:
            # Get index of columns
            column_indices = (
                states.columns.tolist().index("exp_a"),
                states.columns.tolist().index("exp_b"),
                states.columns.tolist().index("edu"),
                states.columns.tolist().index("type"),
                states.columns.tolist().index("choice_lagged"),
                states.columns.tolist().index("emaxs_a"),
                states.columns.tolist().index("emaxs_b"),
                states.columns.tolist().index("emaxs_edu"),
                states.columns.tolist().index("emaxs_home"),
            )

            # Pass states_subset to function as it reduces lookup-time.
            max_col_idx = max(column_indices)
            breakpoint()
            states_subset = (
                states.loc[states.period.eq(period + 1)]
                .iloc[:, : max_col_idx + 1]
                .values
            )

            row_indices = states.loc[states.period.eq(period)].index

            # TODO: Embarrassingly parallel.
            for row_idx in row_indices:

                # Unpack characteristics of current state
                exp_a, exp_b, edu, type_ = states.loc[
                    row_idx, ["exp_a", "exp_b", "edu", "type"]
                ]
                states.loc[
                    row_idx, ["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"]
                ] = get_emaxs_of_subsequent_period(
                    edu_spec["max"],
                    states_subset,
                    row_idx,
                    column_indices,
                    exp_a,
                    exp_b,
                    edu,
                    type_,
                )

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

        if any_interpolated:
            # Get indicator for interpolation and simulation of states
            is_simulated = get_simulated_indicator(
                num_points_interp, num_states, period, is_debug
            )

            # Constructing the exogenous variable for all states, including the ones
            # where simulation will take place. All information will be used in either
            # the construction of the prediction model or the prediction step.
            states = get_exogenous_variables(
                period, states, shifts, edu_spec, optim_paras
            )

            # Constructing the dependent variables for all states at the random subset
            # of points where the EMAX is actually calculated.
            states = get_endogenous_variable(
                period,
                states,
                is_simulated,
                num_draws_emax,
                draws_emax_risk,
                edu_spec,
                optim_paras,
            )

            # Create prediction model based on the random subset of points where the
            # EMAX is actually simulated and thus dependent and independent variables
            # are available. For the interpolation points, the actual values are used.
            states.loc[states.period.eq(period), "emax"] = get_predictions(
                period, states, is_simulated, file_sim, is_write
            )

        else:

            states.loc[states.period.eq(period), "emax"] = construct_emax_risk(
                states,
                num_draws_emax,
                period,
                draws_emax_risk,
                edu_spec,
                optim_paras,
            )

    return states


def get_simulated_indicator(num_points_interp, num_states, period, is_debug):
    """ Get the indicator for points of interpolation and simulation.
    """
    # Drawing random interpolation points
    interpolation_points = np.random.choice(
        range(num_states), size=num_points_interp, replace=False
    )

    # Constructing an indicator whether a state will be simulated or interpolated.
    is_simulated = np.tile(False, num_states)
    is_simulated[interpolation_points] = True

    # Check for debugging cases.
    is_standardized = is_debug and os.path.exists(".interpolation.respy.test")
    if is_standardized:
        with open(".interpolation.respy.test", "r") as file_:
            indicators = []
            for line in file_:
                indicators += [(shlex.split(line)[period] == "True")]
        is_simulated = indicators[:num_states]

    is_simulated = np.array(is_simulated)

    return is_simulated


def get_exogenous_variables(period, states, draws, edu_spec, optim_paras):
    """ Get exogenous variables for interpolation scheme.

    The unused argument is present to align the interface between the PYTHON and FORTRAN
    implementations.

    Parameters
    ----------
    period : int
        Number of period.
    states : pd.DataFrame
    draws : np.array
    edu_spec : dict
    optim_paras : dict

    Returns
    -------
    states : pd.DataFrame

    """
    # Calculate ex-post rewards
    states.loc[states.period.eq(period), "rewards_ex_post_a"] = (
        states.wage_a * draws[0] + states.rewards_systematic_a - states.wage_a
    )
    states.loc[states.period.eq(period), "rewards_ex_post_b"] = (
        states.wage_b * draws[1] + states.rewards_systematic_b - states.wage_b
    )
    states.loc[states.period.eq(period), "rewards_ex_post_edu"] = (
        states.rewards_systematic_edu + draws[2]
    )
    states.loc[states.period.eq(period), "rewards_ex_post_home"] = (
        states.rewards_systematic_home + draws[3]
    )

    # Calculate total utilities
    states.loc[states.period.eq(period), "total_values_a"] = (
        states.rewards_ex_post_a + optim_paras["delta"] * states.emaxs_a
    )
    states.loc[states.period.eq(period), "total_values_b"] = (
        states.rewards_ex_post_b + optim_paras["delta"] * states.emaxs_b
    )
    states.loc[states.period.eq(period), "total_values_edu"] = (
        states.rewards_ex_post_edu + optim_paras["delta"] * states.emaxs_edu
    )
    states.loc[states.period.eq(period), "total_values_home"] = (
        states.rewards_ex_post_home + optim_paras["delta"] * states.emaxs_home
    )

    # Implement level shifts
    states.loc[states.period.eq(period), "maxe"] = states[
        [
            "total_values_a",
            "total_values_b",
            "total_values_edu",
            "total_values_home",
        ]
    ].max(axis=1)

    states.loc[states.period.eq(period), "exogenous_a"] = (
        states.maxe - states.total_values_a
    )
    states.loc[states.period.eq(period), "exogenous_b"] = (
        states.maxe - states.total_values_b
    )
    states.loc[states.period.eq(period), "exogenous_edu"] = (
        states.maxe - states.total_values_edu
    )
    states.loc[states.period.eq(period), "exogenous_home"] = (
        states.maxe - states.total_values_home
    )

    return states


def get_endogenous_variable(
    period,
    states,
    is_simulated,
    num_draws_emax,
    draws_emax_risk,
    edu_spec,
    optim_paras,
):
    """ Construct endogenous variable for the subset of interpolation points.

    TODO: There are a number of states randomly chosen for interpolation which do not
    need calculation. Maybe one can enhance performance by dropping the cases.
    Currently, they are part of the calculation and are skipped in
    :func:`get_predictions`.

    """

    # Simulate the expected future value.
    states.loc[states.period.eq(period), "emax"] = construct_emax_risk(
        states, num_draws_emax, period, draws_emax_risk, edu_spec, optim_paras
    )

    # Construct dependent variable
    states.loc[states.period.eq(period), "endog_variable"] = (
        states.emax - states.maxe
    )

    return states


def get_predictions(period, states, is_simulated, file_sim, is_write):
    """ Fit an OLS regression of the exogenous variables on the endogenous variables and
    use the results to predict the endogenous variables for all points in the state
    space.

    """
    endogenous = states.loc[states.period.eq(period), "endog_variable"].values

    exogenous = states.loc[
        states.period.eq(period),
        ["exogenous_a", "exogenous_b", "exogenous_edu", "exogenous_home"],
    ].values
    # TODO: In https://github.com/OpenSourceEconomics/respy/blob/
    # ed6361d4aed5da6beb7109424c7c506805b97512/respy/python/solve/
    # solve_auxiliary.py#L459, it is stated that infinite values should be replaced, but
    # it was never implemented.
    exogenous = np.hstack(
        (np.ones((exogenous.shape[0], 1)), exogenous, np.sqrt(exogenous))
    )

    # Define ordinary least squares model and fit to the data.
    model = sm.OLS(endogenous[is_simulated], exogenous[is_simulated])
    results = model.fit()

    # Use the model to predict EMAX for all states. As in Keane & Wolpin (1994),
    # negative predictions are truncated to zero.
    endogenous_predicted = results.predict(exogenous)
    endogenous_predicted = np.clip(endogenous_predicted, 0.00, None)

    # Construct predicted EMAX for all states and the replace interpolation points with
    # simulated values.
    predictions = (
        endogenous_predicted
        + states.loc[states.period.eq(period), "maxe"].values
    )
    predictions[is_simulated] = (
        endogenous[is_simulated]
        + states.loc[states.period.eq(period), "maxe"].values[is_simulated]
    )

    check_prediction_model(endogenous_predicted, model)

    # Write out some basic information to spot problems easily.
    if is_write:
        record_solution_prediction(results, file_sim)

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


def calculate_wages_systematic(states, optim_paras):
    """Calculate systematic wages.

    Parameters
    ----------
    states : pd.DataFrame
        DataFrame containing covariates and rewards.
    optim_paras : dict

    Returns
    -------
    states : pd.DataFrame
        DataFrame with calculated systematic wages.

    """
    states["exp_a_sq"] = states.exp_a ** 2 / 100
    states["exp_b_sq"] = states.exp_b ** 2 / 100

    relevant_covariates = [
        "intercept",
        "edu",
        "exp_a",
        "exp_a_sq",
        "exp_b",
        "exp_b_sq",
        "hs_graduate",
        "co_graduate",
        "period",
        "is_minor",
    ]

    # Calculate systematic part of wages in OCCUPATION A and OCCUPATION B
    states["wage_a"] = states[
        relevant_covariates + ["any_exp_a", "work_a_lagged"]
    ].dot(optim_paras["coeffs_a"][:12])
    states.wage_a = np.clip(np.exp(states.wage_a), 0.0, HUGE_FLOAT)

    states["wage_b"] = states[
        relevant_covariates + ["any_exp_b", "work_b_lagged"]
    ].dot(optim_paras["coeffs_b"][:12])
    states.wage_b = np.clip(np.exp(states.wage_b), 0.0, HUGE_FLOAT)

    # We need to add the type-specific deviations here as these are part of
    # skill-function component.
    types_dummy = pd.get_dummies(states.type, prefix="type")
    type_deviations = types_dummy.dot(optim_paras["type_shifts"][:, :2])

    states.wage_a = states.wage_a * np.exp(type_deviations.iloc[:, 0])
    states.wage_b = states.wage_b * np.exp(type_deviations.iloc[:, 1])

    return states
