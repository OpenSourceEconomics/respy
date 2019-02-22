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
from respy.python.shared.shared_auxiliary import get_continuation_value
import pandas as pd
from respy.custom_exceptions import InadmissibleStateError
from respy.python.shared.shared_auxiliary import create_covariates
from numba import njit


@njit
def pyth_create_state_space(num_periods, num_types, edu_starts, edu_max):
    """ Create the state space.

    The state space consists of all admissible combinations of the following elements:
    period, experience in occupation A, experience in occupation B, years of schooling,
    the lagged choice and the type of the agent.

    The major problem which is solved here is that the natural representation of the
    state space is a graph where each parent node is connected with its child nodes.
    This structure makes it easy to traverse the graph. In contrast to that, for many
    subsequent calculations, e.g. generating the covariates for each state, a denser
    data format has better performance, but looses the information on the connections.

    The current implementation of :data:`states` and :data:`states_indexer` allows to
    have both advantages at the cost of an additional object. :data:`states` stores the
    information on states in a relational format. :data:`states_indexer` is a matrix
    where each characteristic of the state space represents one dimension. The values of
    the matrix are the indices of states in :data:`states`. Traversing the state space
    is as easy as incrementing the right indices of :data:`states_indexer` by 1 and use
    the resulting index in :data:`states`.

    Parameters
    ----------
    num_periods : int
        Number of periods. ???
    num_types : int
        Number of types. ???
    edu_starts : List[int]
        Contains values of initial education.
    edu_max : int
        Maximum of achievable education.

    Returns
    -------
    states : pd.DataFrame
        This DataFrame contains all admissible states.
    states_indexer : np.array
        A matrix where each dimension represents a characteristic of the state space.
        Switching from one state is possible via incrementing appropriate indices by 1.

    Examples
    --------
    >>> num_periods = 40
    >>> num_types = 1
    >>> edu_starts, edu_max = [10], 20
    >>> states, indexer = pyth_create_state_space(
    ...     num_periods, num_types, edu_starts, edu_max
    ... )
    >>> states.shape
    (324263, 6)
    >>> indexer.shape
    (40, 40, 40, 21, 4, 1)

    """
    data = []

    # Initialize the state indexer object which enables faster lookup of states in the
    # pandas DataFrame. The dimensions of the matrix are the characteristics of the
    # agents and the value is the index in the DataFrame.
    shape = (num_periods, num_periods, num_periods, edu_max + 1, 4, num_types)
    states_indexer = np.full(shape, -1, dtype=np.int32)

    # Initialize counter
    i = 0

    # Construct state space by periods
    for period in range(num_periods):

        # Loop over all unobserved types
        for type_ in range(num_types):

            # Loop overall all initial levels of schooling
            for edu_start in edu_starts:

                # For occupations and education it is necessary to loop over period
                # + 1 as zero has to be included if it is never this choice and period
                # + 1 if it is always the same choice.

                # Furthermore, the time constraint of agents is always fulfilled as
                # previous choices limit the space for subsequent choices.

                # Loop over all admissible work experiences for Occupation A
                for exp_a in range(period + 1):

                    # Loop over all admissible work experience for Occupation B
                    for exp_b in range(period + 1 - exp_a):

                        # Loop over all admissible additional education levels
                        for edu_add in range(
                            min(
                                period + 1 - exp_a - exp_b,
                                edu_max + 1 - edu_start,
                            )
                        ):

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

                                # (2, 1) An individual that has never worked in
                                # Occupation A cannot have that lagged activity.
                                if (choice_lagged == 1) and (exp_a == 0):
                                    continue

                                # (3, 1) An individual that has never worked in
                                # Occupation B cannot have a that lagged activity.
                                if (choice_lagged == 2) and (exp_b == 0):
                                    continue

                                # (1, 1) In the first period individual either were in
                                # school the previous period as well or at home. They
                                # cannot have any work experience.
                                if period == 0:
                                    if choice_lagged in [1, 2]:
                                        continue

                                # Collect mapping of state space to array index.
                                states_indexer[
                                    period,
                                    exp_a,
                                    exp_b,
                                    edu_start + edu_add,
                                    choice_lagged - 1,
                                    type_,
                                ] = i

                                i += 1

                                # Store information in a dictionary and append to data.
                                row = (
                                    period,
                                    exp_a,
                                    exp_b,
                                    edu_start + edu_add,
                                    choice_lagged,
                                    type_,
                                )
                                data.append(row)

    states = np.array(data)

    return states, states_indexer


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
    state_space,
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
    state_space
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
    states_per_period = state_space.states_per_period

    if is_myopic:
        record_solution_progress(-2, file_sim)

        state_space.states["emax"] = 0.0
        state_space.states["emaxs_a"] = 0.0
        state_space.states["emaxs_b"] = 0.0
        state_space.states["emaxs_edu"] = 0.0
        state_space.states["emaxs_home"] = 0.0

        return state_space

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
        if period == num_periods - 1:
            # Has to be a loop because columns are not initialized.
            for col in ["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"]:
                state_space.states.loc[
                    state_space.states.period.eq(period), col
                ] = 0.0
        else:
            row_indices = state_space.states.loc[
                state_space.states.period.eq(period)
            ].index

            # TODO: Embarrassingly parallel.
            for row_idx in row_indices:

                # Unpack characteristics of current state
                exp_a, exp_b, edu, type_ = state_space.states.loc[
                    row_idx, ["exp_a", "exp_b", "edu", "type"]
                ]

                state_space.states.loc[
                    row_idx, ["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"]
                ] = get_emaxs_of_subsequent_period(
                    edu_spec["max"],
                    state_space,
                    row_idx,
                    period,
                    int(exp_a),
                    int(exp_b),
                    int(edu),
                    int(type_),
                )

        # Extract auxiliary objects
        draws_emax_standard = periods_draws_emax[period, :, :]
        num_states = states_per_period[period]

        # Treatment of the disturbances for the risk-only case is straightforward. Their
        # distribution is fixed once and for all.
        draws_emax_risk = transform_disturbances(
            draws_emax_standard,
            np.full(4, 0.0),
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
            state_space.states = get_exogenous_variables(
                period, state_space.states, shifts, edu_spec, optim_paras
            )

            # Constructing the dependent variables for all states at the random subset
            # of points where the EMAX is actually calculated.
            state_space.states = get_endogenous_variable(
                period,
                state_space.states,
                is_simulated,
                num_draws_emax,
                draws_emax_risk,
                edu_spec,
                optim_paras,
            )

            # Create prediction model based on the random subset of points where the
            # EMAX is actually simulated and thus dependent and independent variables
            # are available. For the interpolation points, the actual values are used.
            state_space.states.loc[
                state_space.states.period.eq(period), "emax"
            ] = get_predictions(
                period, state_space.states, is_simulated, file_sim, is_write
            )

        else:

            state_space.states.loc[
                state_space.states.period.eq(period), "emax"
            ] = construct_emax_risk(
                state_space.states.loc[
                    state_space.states.period.eq(period)
                ].copy(),
                draws_emax_risk,
                optim_paras,
            )

    return state_space


def get_simulated_indicator(num_points_interp, num_states, period, is_debug):
    """ Get the indicator for points of interpolation and simulation.

    Parameters
    ----------
    num_points_interp : int
        Number of states which will be interpolated.
    num_states : int
        Number of states.
    period : int
        Number of period.
    is_debug : bool
        Flag for debugging. If true, interpolation points are taken from file.

    Returns
    -------
    is_simulated : np.array
        Array of shape (num_states) indicating states which will be interpolated.

    """
    # Drawing random interpolation points
    interpolation_points = np.random.choice(
        range(num_states), size=num_points_interp, replace=False
    )

    # Constructing an indicator whether a state will be simulated or interpolated.
    is_simulated = np.full(num_states, False)
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
    total_values, rewards_ex_post = get_continuation_value(
        states[["wage_a", "wage_b"]].values,
        states[
            [
                "rewards_systematic_a",
                "rewards_systematic_b",
                "rewards_systematic_edu",
                "rewards_systematic_home",
            ]
        ].values,
        draws.reshape(1, -1),
        states[["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"]].values,
        optim_paras["delta"],
    )

    states[
        [
            "rewards_ex_post_a",
            "rewards_ex_post_b",
            "rewards_ex_post_edu",
            "rewards_ex_post_home",
        ]
    ] = rewards_ex_post
    states[
        [
            "total_values_a",
            "total_values_b",
            "total_values_edu",
            "total_values_home",
        ]
    ] = total_values

    # Implement level shifts
    states.loc[states.period.eq(period), "max_emax"] = states[
        [
            "total_values_a",
            "total_values_b",
            "total_values_edu",
            "total_values_home",
        ]
    ].max(axis=1)

    states.loc[states.period.eq(period), "exogenous_a"] = (
        states.max_emax - states.total_values_a
    )
    states.loc[states.period.eq(period), "exogenous_b"] = (
        states.max_emax - states.total_values_b
    )
    states.loc[states.period.eq(period), "exogenous_edu"] = (
        states.max_emax - states.total_values_edu
    )
    states.loc[states.period.eq(period), "exogenous_home"] = (
        states.max_emax - states.total_values_home
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
        states.loc[states.period.eq(period)], draws_emax_risk, optim_paras
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

    # TODO: Is this still necessary?
    if os.path.exists(".restud.respy.scratch"):
        states["exp_a_sq"] *= 100.00
        states["exp_b_sq"] *= 100.00

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


class StateSpace:
    """This class represents a state space.

    Attributes
    ----------
    states : pd.DataFrame
    indexer : np.array

    """

    def create_state_space(self, *args):
        """This function creates the state space.

        It should return two objects. First, a DataFrame where each row represents one
        state. This object captures characteristics of states in a dense format. The
        second object is a Numpy matrix where each dimension represents one
        characteristic of the state and the values contain the indices of the state in
        the DataFrame.

        """
        data, self.indexer = pyth_create_state_space(*args)
        self.states = pd.DataFrame(
            data,
            columns=[
                "period",
                "exp_a",
                "exp_b",
                "edu",
                "choice_lagged",
                "type",
            ],
        )

    def create_covariates(self):
        self.states = create_covariates(self.states)

    @property
    def states_per_period(self):
        return self.states.groupby("period").count().iloc[:, 0].values

    def _get_fortran_counterparts(self):
        periods_rewards_systematic = np.full(
            (self.states_per_period.shape[0], self.states_per_period.max(), 4),
            np.nan,
        )
        for period, group in self.states.groupby("period"):
            sub = group[
                [
                    "rewards_systematic_a",
                    "rewards_systematic_b",
                    "rewards_systematic_edu",
                    "rewards_systematic_home",
                ]
            ].values

            periods_rewards_systematic[period, : sub.shape[0]] = sub

        periods_emax = np.full(
            (self.states_per_period.shape[0], self.states_per_period.max(), 4),
            np.nan,
        )
        for period, group in self.states.groupby("period"):
            sub = group[
                ["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"]
            ].values

            periods_emax[period, : sub.shape[0], :] = sub

        mapping_state_idx = self.indexer
        states_all = None

        return (
            periods_rewards_systematic,
            self.states_per_period,
            mapping_state_idx,
            periods_emax,
            states_all,
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, key):
        """

        Raises
        ------
        InadmissibleStateError
            The error is raised if a non-valid indexer is used. The simplest way to
            raise this error is to assign -1 to every inadmissible state. This value is
            still an integer to keep memory costs down.

        """
        key = tuple(int(i) for i in key)
        position = self.indexer[key]

        if position == -1:
            raise InadmissibleStateError(
                ", ".join(
                    [
                        "{}: {}".format(k, v)
                        for k, v in zip(self.indexer_dim_names, key)
                    ]
                )
            )
        else:
            return self.states.iloc[position]
