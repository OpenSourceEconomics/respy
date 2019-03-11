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
from respy.custom_exceptions import StateSpaceError
from respy.python.shared.shared_auxiliary import create_covariates
from numba import njit
from respy.python.shared.shared_constants import MISSING_FLOAT, MISSING_INT


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

                                # Continue if state still exist. This is only caused by
                                # multiple starting values of education.
                                if (
                                    states_indexer[
                                        period,
                                        exp_a,
                                        exp_b,
                                        edu_start + edu_add,
                                        choice_lagged - 1,
                                        type_,
                                    ]
                                    != -1
                                ):
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


def pyth_calculate_rewards_systematic(states, covariates, optim_paras):
    """ Calculate systematic rewards.

    Parameters
    ----------
    states : np.ndarray
    covariates : np.ndarray
    optim_paras : dict

    """
    # Calculate common and general rewards component.
    rewards_general = calculate_rewards_general(
        covariates, optim_paras["coeffs_a"][12:], optim_paras["coeffs_b"][12:]
    )
    rewards_common = calculate_rewards_common(
        covariates, optim_paras["coeffs_common"]
    )

    # Calculate the systematic part of OCCUPATION A and OCCUPATION B rewards.
    # These are defined in a general sense, where not only wages matter.
    wages = calculate_wages_systematic(
        states,
        covariates,
        optim_paras["coeffs_a"][:12],
        optim_paras["coeffs_b"][:12],
        optim_paras["type_shifts"][:, :2],
    )

    rewards = np.full((states.shape[0], 4), np.nan)

    rewards[:, :2] = wages + rewards_general

    # Calculate systematic part of SCHOOL rewards
    covariates_education = np.c_[
        np.ones(states.shape[0]),
        covariates[:, 9:13],
        states[:, 0],
        covariates[:, 13],
    ]

    rewards[:, 2] = covariates_education.dot(optim_paras["coeffs_edu"])

    # Calculate systematic part of HOME
    covariates_home = np.c_[np.ones(states.shape[0]), covariates[:, 14:]]

    rewards[:, 3] = covariates_home.dot(optim_paras["coeffs_home"])

    # Now we add the type-specific deviation for SCHOOL and HOME.
    type_dummies = get_dummies(states[:, 5])
    type_deviations = type_dummies.dot(optim_paras["type_shifts"][:, 2:])

    rewards[:, 2:] = rewards[:, 2:] + type_deviations

    # We can now also added the common component of rewards.
    rewards_systematic = rewards + rewards_common

    return rewards_systematic, rewards_general, rewards_common, wages


def pyth_backward_induction(
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
    state_space.emaxs = np.zeros((state_space.states.shape[0], 5))

    if is_myopic:
        record_solution_progress(-2, file_sim)

        return state_space

    # Construct auxiliary objects
    shocks_cov = optim_paras["shocks_cholesky"].dot(
        optim_paras["shocks_cholesky"].T
    )

    # Auxiliary objects. These shifts are used to determine the expected values of the
    # two labor market alternatives. These are log normal distributed and thus the draws
    # cannot simply set to zero.
    shifts = np.zeros(4)
    shifts[:2] = np.clip(np.diag(shocks_cov)[:2] / 2.0, 0.0, HUGE_FLOAT)

    for period in reversed(range(state_space.num_periods)):

        if period == state_space.num_periods - 1:
            pass

        else:
            states_period = state_space.get_attribute_from_period(
                "states", period
            )

            state_space.emaxs = get_emaxs_of_subsequent_period(
                states_period,
                state_space.indexer,
                state_space.emaxs,
                edu_spec["max"],
            )

        # Extract auxiliary objects
        draws_emax_standard = periods_draws_emax[period, :, :]
        num_states = state_space.states_per_period[period]

        # Treatment of the disturbances for the risk-only case is straightforward. Their
        # distribution is fixed once and for all.
        draws_emax_risk = transform_disturbances(
            draws_emax_standard, np.zeros(4), optim_paras["shocks_cholesky"]
        )

        if is_write:
            record_solution_progress(4, file_sim, period, num_states)

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In that case no interpolation is needed.
        any_interpolated = (
            num_points_interp <= num_states
        ) and is_interpolated

        # Unpack necessary attributes
        rewards_period = state_space.get_attribute_from_period(
            "rewards", period
        )
        emaxs_period = state_space.get_attribute_from_period("emaxs", period)[
            :, :4
        ]

        if any_interpolated:
            # Get indicator for interpolation and simulation of states
            is_simulated = get_simulated_indicator(
                num_points_interp, num_states, period, is_debug
            )

            # Constructing the exogenous variable for all states, including the ones
            # where simulation will take place. All information will be used in either
            # the construction of the prediction model or the prediction step.
            exogenous, max_emax = get_exogenous_variables(
                rewards_period, emaxs_period, shifts, edu_spec, optim_paras
            )

            # Constructing the dependent variables for all states at the random subset
            # of points where the EMAX is actually calculated.
            endogenous = get_endogenous_variable(
                rewards_period,
                emaxs_period,
                max_emax,
                is_simulated,
                num_draws_emax,
                draws_emax_risk,
                edu_spec,
                optim_paras,
            )

            # Create prediction model based on the random subset of points where the
            # EMAX is actually simulated and thus dependent and independent variables
            # are available. For the interpolation points, the actual values are used.
            emax = get_predictions(
                endogenous,
                exogenous,
                max_emax,
                is_simulated,
                file_sim,
                is_write,
            )

        else:

            emax = construct_emax_risk(
                rewards_period, emaxs_period, draws_emax_risk, optim_paras
            )

        state_space.get_attribute_from_period("emaxs", period)[:, 4] = emax

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


def get_exogenous_variables(rewards, emaxs, draws, edu_spec, optim_paras):
    """ Get exogenous variables for interpolation scheme.

    The unused argument is present to align the interface between the PYTHON and FORTRAN
    implementations.

    Parameters
    ----------
    rewards : np.ndarray
    emaxs : np.ndarray
        Array with shape (num_states_in_period, 4)
    draws : np.ndarray
    edu_spec : dict
    optim_paras : dict

    Returns
    -------
    states : pd.DataFrame

    """
    total_values, _ = get_continuation_value(
        rewards[:, -2:],
        rewards[:, :4],
        draws.reshape(1, -1),
        emaxs,
        optim_paras["delta"],
    )

    max_emax = total_values.max(axis=1)

    exogenous = max_emax - total_values.reshape(-1, 4)

    return exogenous, max_emax.reshape(-1)


def get_endogenous_variable(
    rewards,
    emaxs,
    max_emax,
    is_simulated,
    num_draws_emax,
    draws_emax_risk,
    edu_spec,
    optim_paras,
):
    """ Construct endogenous variable for the subset of interpolation points.

    Parameters
    ----------
    rewards : np.ndarray
        Array with shape (num_states_in_period, 9)
    emaxs : np.ndarray
        Array with shape (num_states_in_period, 4).
    max_emax : np.ndarray
        Array with shape (num_states_in_period,) containing maximum of exogenous emax.

    """
    emax = construct_emax_risk(rewards, emaxs, draws_emax_risk, optim_paras)

    endogenous = emax - max_emax

    endogenous[~is_simulated] = np.nan

    return endogenous


def get_predictions(
    endogenous, exogenous, max_emax, is_simulated, file_sim, is_write
):
    """ Fit an OLS regression of the exogenous variables on the endogenous variables and
    use the results to predict the endogenous variables for all points in the state
    space.

    """
    exogenous = np.c_[
        np.ones(exogenous.shape[0]), exogenous, np.sqrt(exogenous)
    ]

    # Define ordinary least squares model and fit to the data.
    model = sm.OLS(endogenous[is_simulated], exogenous[is_simulated])
    results = model.fit()

    # Use the model to predict EMAX for all states. As in Keane & Wolpin (1994),
    # negative predictions are truncated to zero.
    endogenous_predicted = results.predict(exogenous)
    endogenous_predicted = np.clip(endogenous_predicted, 0.00, None)

    # Construct predicted EMAX for all states and the replace interpolation points with
    # simulated values.
    predictions = endogenous_predicted + max_emax
    predictions[is_simulated] = (
        endogenous[is_simulated] + max_emax[is_simulated]
    )

    check_prediction_model(endogenous_predicted, model)

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

    return True


def calculate_wages_systematic(
    states, covariates, coeffs_a, coeffs_b, type_shifts
):
    """Calculate systematic wages.

    Parameters
    ----------
    states : np.ndarray
        Array with shape (num_states, 5) containing state information.
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates.
    coeffs_a : np.ndarray
        Array with shape (12) containing coefficients of occupation A.
    coeffs_b : np.ndarray
        Array with shape (12) containing coefficients of occupation B.
    type_shifts : np.ndarray
        Array with shape (2, 2).

    Returns
    -------
    wages : np.ndarray
        Array with shape (num_states, 2) containing systematic wages.

    Example
    -------
    >>> np.random.seed(42)
    >>> from respy.python.solve.solve_auxiliary import StateSpace
    >>> num_types = 2
    >>> state_space = StateSpace(1, num_types, [12, 16], 20)
    >>> coeffs_a = np.random.randint(0, 101, 12) / 100
    >>> coeffs_b = np.random.randint(0, 101, 12) / 100
    >>> type_shifts = np.random.randn(num_types, 4)
    >>> calculate_wages_systematic(
    ...     state_space.states,
    ...     state_space.covariates,
    ...     coeffs_a,
    ...     coeffs_b,
    ...     type_shifts,
    ... )
    array([[2.13158868e+06, 1.87035803e+01],
           [2.13158868e+06, 1.87035803e+01],
           [1.99710249e+08, 2.93330530e+01],
           [1.99710249e+08, 2.93330530e+01],
           [3.84214409e+05, 3.40802479e+00],
           [3.84214409e+05, 3.40802479e+00],
           [3.59973554e+07, 5.34484681e+00],
           [3.59973554e+07, 5.34484681e+00]])

    """
    exp_a_sq = states[:, 1] ** 2 / 100
    exp_b_sq = states[:, 2] ** 2 / 100

    if os.path.exists(".restud.respy.scratch"):
        exp_a_sq *= 100.00
        exp_b_sq *= 100.00

    relevant_covariates = np.c_[
        np.ones(states.shape[0]),
        states[:, [3, 1]],
        exp_a_sq,
        states[:, 2],
        exp_b_sq,
        covariates[:, 9:11],
        states[:, 0],
        covariates[:, 13],
    ]

    # Calculate systematic part of wages in OCCUPATION A and OCCUPATION B
    wages = np.full((states.shape[0], 2), np.nan)

    wages[:, 0] = np.c_[relevant_covariates, covariates[:, [7, 2]]].dot(
        coeffs_a
    )

    wages[:, 1] = np.c_[relevant_covariates, covariates[:, [8, 3]]].dot(
        coeffs_b
    )

    wages = np.clip(np.exp(wages), 0.0, HUGE_FLOAT)

    # We need to add the type-specific deviations here as these are part of
    # skill-function component.
    type_dummies = get_dummies(states[:, 5])
    type_deviations = type_dummies.dot(type_shifts[:, :2])

    wages = wages * np.exp(type_deviations)

    return wages


@njit
def get_dummies(a):
    """Create dummy matrix from array with indicators.

    Note that, the indicators need to be counting from zero onwards.

    Paramters
    ---------
    a : np.array
        Array with shape (n) containing k different indicators from 0 to k-1.

    Returns
    -------
    b : np.ndarray
        Matrix with shape (n, k) where each column is a dummy vector for type i = 0,
        ..., k-1.

    Example
    -------
    >>> a = np.random.randint(0, 6, 100)
    >>> res = get_dummies(a)
    >>> import pandas as pd
    >>> res_pandas = pd.get_dummies(a).values
    >>> assert np.allclose(res, res_pandas)

    """
    n_values = np.max(a) + 1
    return np.eye(n_values)[a]


class StateSpace:
    """This class represents a state space.

    Attributes
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        choice_lagged and type information.
    indexer : np.array
        Array with shape (num_periods, num_periods, num_periods, edu_max, 4, num_types).

    """

    states_columns = [
        "period",
        "exp_a",
        "exp_b",
        "edu",
        "choice_lagged",
        "type",
    ]

    covariates_columns = [
        "not_exp_a_lagged",
        "not_exp_b_lagged",
        "work_a_lagged",
        "work_b_lagged",
        "edu_lagged",
        "not_any_exp_a",
        "not_any_exp_b",
        "any_exp_a",
        "any_exp_b",
        "hs_graduate",
        "co_graduate",
        "is_return_not_high_school",
        "is_return_high_school",
        "is_minor",
        "is_young_adult",
        "is_adult",
    ]

    rewards_columns = [
        "rewards_systematic_a",
        "rewards_systematic_b",
        "rewards_systematic_edu",
        "rewards_systematic_home",
        "rewards_general_a",
        "rewards_general_b",
        "rewards_common",
        "wage_a",
        "wage_b",
    ]

    def __init__(
        self, num_periods, num_types, edu_starts, edu_max, optim_paras=None
    ):
        self.num_periods = num_periods

        self.states, self.indexer = pyth_create_state_space(
            num_periods, num_types, edu_starts, edu_max
        )

        self.covariates = create_covariates(self.states)

        # This measure is related to test_f2py::test_2.
        if optim_paras:

            self.rewards = np.c_[
                pyth_calculate_rewards_systematic(
                    self.states, self.covariates, optim_paras
                )
            ]

        self._create_slices_by_periods(num_periods)

    @property
    def states_per_period(self):
        return np.array(
            [len(range(i.start, i.stop)) for i in self.slices_by_periods]
        )

    def update_systematic_rewards(self, optim_paras):
        self.rewards = np.c_[
            pyth_calculate_rewards_systematic(
                self.states, self.covariates, optim_paras
            )
        ]

    def _create_attributes_from_fortran_counterparts(
        self, periods_emax
    ):
        self.emaxs = np.c_[
            np.zeros((self.states_per_period.sum(), 4)),
            periods_emax[periods_emax != -99],
        ]

    def _get_fortran_counterparts(self):
        try:
            periods_rewards_systematic = np.full(
                (self.num_periods, self.states_per_period.max(), 4),
                MISSING_FLOAT,
            )
            for period in range(self.num_periods):
                rewards = self.get_attribute_from_period("rewards", period)[
                    :, :4
                ]

                periods_rewards_systematic[
                    period, : rewards.shape[0]
                ] = rewards
        except KeyError:
            periods_rewards_systematic = None

        try:
            periods_emax = np.full(
                (self.num_periods, self.states_per_period.max()), MISSING_FLOAT
            )
            for period in range(self.num_periods):
                emax = self.get_attribute_from_period("emaxs", period)[:, 4]

                periods_emax[period, : emax.shape[0]] = emax
        except KeyError:
            periods_emax = None

        states_all = np.full(
            (self.num_periods, self.states_per_period[-1], 5), MISSING_INT
        )
        for period in range(self.num_periods):
            states = self.get_attribute_from_period("states", period)[:, 1:]

            states_all[period, : states.shape[0], :] = states

        # The indexer has to be modified because ``mapping_state_idx`` resets the
        # counter to zero for each period and ``self.indexer`` not. For every period,
        # subtract the minimum index.
        mapping_state_idx = self.indexer.copy()
        for period in range(self.num_periods):
            mask = mapping_state_idx[period] != -1
            minimum_index = mapping_state_idx[period][mask].min()

            mapping_state_idx[period][mask] -= minimum_index

        mapping_state_idx[mapping_state_idx == -1] = MISSING_INT

        return (
            states_all,
            mapping_state_idx,
            periods_rewards_systematic,
            periods_emax,
        )

    def get_attribute_from_period(self, attr, period):
        try:
            attribute = getattr(self, attr)
        except AttributeError:
            raise StateSpaceError("Inadmissible attribute.")

        try:
            indices = self.slices_by_periods[period]
        except IndexError:
            raise StateSpaceError("Inadmissible period.")

        return attribute[indices]

    def _create_slices_by_periods(self, num_periods):
        self.slices_by_periods = []
        for i in range(num_periods):
            idx_start, idx_end = np.where(self.states[:, 0] == i)[0][[0, -1]]
            self.slices_by_periods.append(slice(idx_start, idx_end + 1))

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
        position = self.indexer[key]

        if position == -1:
            raise StateSpaceError(
                "Inadmissible state with "
                + ", ".join(
                    [
                        "{}: {}".format(k, v)
                        for k, v in zip(self.states_columns, key)
                    ]
                )
            )
        else:
            return self.states[position]
