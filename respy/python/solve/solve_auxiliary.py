import os
import shlex

import numpy as np
import pandas as pd
from numba import njit

from respy.custom_exceptions import StateSpaceError
from respy.python.shared.shared_auxiliary import calculate_rewards_common
from respy.python.shared.shared_auxiliary import calculate_rewards_general
from respy.python.shared.shared_auxiliary import create_covariates
from respy.python.shared.shared_auxiliary import get_continuation_value
from respy.python.shared.shared_auxiliary import get_emaxs_of_subsequent_period
from respy.python.shared.shared_auxiliary import ols
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import KW_SQUARED_EXPERIENCES
from respy.python.solve.solve_risk import construct_emax_risk


@njit
def pyth_create_state_space(num_periods, num_types, edu_starts, edu_max):
    """Create the state space.

    The state space consists of all admissible combinations of the following elements:
    period, experience in OCCUPATION A, experience in OCCUPATION B, years of schooling,
    the lagged choice and the type of the agent.

    The major problem which is solved here is that the natural representation of the
    state space is a graph where each parent node is connected with its children's
    nodes. This structure makes it easy to traverse the graph. In contrast to that, for
    many subsequent calculations, e.g. generating the covariates for each state, a
    tabular format has better performance, but looses the information on the
    connections.

    The current implementation of ``states`` and ``indexer`` allows to have both
    advantages at the cost of an additional object. ``states`` stores the information on
    states in a tabular format. ``indexer`` is a matrix where each characteristic of the
    state space represents one dimension. The values of the matrix are the indices of
    states in ``states``. Traversing the state space is as easy as incrementing the
    right indices of ``indexer`` by 1 and use the resulting index in ``states``.

    Parameters
    ----------
    num_periods : int
        Number of periods in the state space.
    num_types : int
        Number of types of agents.
    edu_starts : List[int]
        Contains levels of initial education.
    edu_max : int
        Maximum level of education which can be obtained by an agent.

    Returns
    -------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, experience in OCCUPATION A,
        experience in OCCUPATION B, years of schooling, the lagged choice and the type
        of the agent.
    indexer : np.ndarray
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
    (317367, 6)
    >>> indexer.shape
    (40, 40, 40, 21, 4, 1)

    """
    data = []

    shape = (num_periods, num_periods, num_periods, edu_max + 1, 4, num_types)
    indexer = np.full(shape, -1, dtype=np.int32)

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
                            min(period + 1 - exp_a - exp_b, edu_max + 1 - edu_start)
                        ):

                            # Loop over all admissible values for the lagged activity:
                            # (1) Occupation A, (2) Occupation B, (3) Education, and (4)
                            # Home.
                            for choice_lagged in [1, 2, 3, 4]:

                                if period > 0:

                                    # (0, 1) Whenever an agent has only worked in
                                    # Occupation A, then the lagged choice cannot be
                                    # anything other than one.
                                    if choice_lagged != 1 and exp_a == period:
                                        continue

                                    # (0, 2) Whenever an agent has only worked in
                                    # Occupation B, then the lagged choice cannot be
                                    # anything other than two
                                    if choice_lagged != 2 and exp_b == period:
                                        continue

                                    # (0, 3) Whenever an agent has only acquired
                                    # additional education, then the lagged choice
                                    # cannot be anything other than three.
                                    if choice_lagged != 3 and edu_add == period:
                                        continue

                                    # (0, 4) Whenever an agent has not acquired any
                                    # additional education and we are not in the first
                                    # period, then lagged activity cannot take a value
                                    # of three.
                                    if choice_lagged == 3 and edu_add == 0:
                                        continue

                                    # (0, 5) Whenever an agent has always chosen
                                    # Occupation A, Occupation B or education, then
                                    # lagged activity cannot take a value of four.
                                    if (
                                        choice_lagged == 4
                                        and exp_a + exp_b + edu_add == period
                                    ):
                                        continue

                                # (2, 1) An individual that has never worked in
                                # Occupation A cannot have that lagged activity.
                                if choice_lagged == 1 and exp_a == 0:
                                    continue

                                # (3, 1) An individual that has never worked in
                                # Occupation B cannot have a that lagged activity.
                                if choice_lagged == 2 and exp_b == 0:
                                    continue

                                # (1, 1) In the first period individual either were in
                                # school the previous period as well or at home. They
                                # cannot have any work experience.
                                if period == 0:
                                    if choice_lagged in [1, 2]:
                                        continue

                                # Continue if state still exist. This condition is only
                                # triggered by multiple initial levels of education.
                                if (
                                    indexer[
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
                                indexer[
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

    return states, indexer


def pyth_calculate_rewards_systematic(states, covariates, optim_paras):
    """Calculate systematic rewards for each state.

    Parameters
    ----------
    states : np.ndarray
        Array with shape (num_states, 6).
    covariates : np.ndarray
        Array with shape (num_states, 16).
    optim_paras : dict
        Contains parameters affected by the optimization.

    """
    # Calculate common and general rewards component.
    rewards_general = calculate_rewards_general(
        covariates, optim_paras["coeffs_a"][12:], optim_paras["coeffs_b"][12:]
    )
    rewards_common = calculate_rewards_common(covariates, optim_paras["coeffs_common"])

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
    covariates_education = np.column_stack(
        (np.ones(states.shape[0]), covariates[:, 9:13], states[:, 0], covariates[:, 13])
    )

    rewards[:, 2] = covariates_education.dot(optim_paras["coeffs_edu"])

    # Calculate systematic part of HOME
    covariates_home = np.column_stack((np.ones(states.shape[0]), covariates[:, 14:]))

    rewards[:, 3] = covariates_home.dot(optim_paras["coeffs_home"])

    # Add the type-specific deviation for SCHOOL and HOME.
    type_dummies = get_dummies(states[:, 5])
    type_deviations = type_dummies.dot(optim_paras["type_shifts"][:, 2:])

    rewards[:, 2:] = rewards[:, 2:] + type_deviations

    rewards_systematic = rewards + rewards_common

    return rewards_systematic, rewards_general, rewards_common, wages


def pyth_backward_induction(
    periods_draws_emax,
    state_space,
    is_debug,
    is_interpolated,
    num_points_interp,
    optim_paras,
):
    """ Calculate utilities with backward induction.

    Parameters
    ----------
    periods_draws_emax : np.ndarray
        Array with shape (num_periods, num_draws, num_choices) containing the
        random draws used to simulate the emax.
    state_space : class
        State space object.
    is_debug : bool
        Flag for debug modus.
    is_interpolated : np.array
        Flag indicating whether interpolation is used to construct the emax in a period.
    num_points_interp : int
        Number of states for which the emax will be interpolated.
    optim_paras : dict
        Parameters affected by optimization.

    Returns
    -------
    state_space : class
        State space containing the emax of the subsequent period of each choice, columns
        0-3, as well as the maximum emax of the current period for each state, column 4,
        in ``state_space.emaxs``.

    """
    state_space.emaxs = np.zeros((state_space.num_states, 5))

    # For myopic agents, utility of later periods does not play a role.
    if optim_paras["delta"] == 0:
        return state_space

    # Unpack arguments.
    delta = optim_paras["delta"]
    shocks_cholesky = optim_paras["shocks_cholesky"]

    shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)

    # These shifts are used to determine the expected values of the two labor market
    # alternatives. These are log normal distributed and thus the draws cannot simply
    # set to zero.
    shifts = np.zeros(4)
    shifts[:2] = np.clip(np.exp(np.diag(shocks_cov)[:2] / 2.0), 0.0, HUGE_FLOAT)

    for period in reversed(range(state_space.num_periods)):

        if period == state_space.num_periods - 1:
            pass

        else:
            states_period = state_space.get_attribute_from_period("states", period)

            state_space.emaxs = get_emaxs_of_subsequent_period(
                states_period,
                state_space.indexer,
                state_space.emaxs,
                state_space.edu_max,
            )

        num_states = state_space.states_per_period[period]

        # Treatment of the disturbances for the risk-only case is straightforward. Their
        # distribution is fixed once and for all.
        draws_emax_standard = periods_draws_emax[period]
        draws_emax_risk = transform_disturbances(
            draws_emax_standard, np.zeros(4), shocks_cholesky
        )

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In that case no interpolation is needed.
        any_interpolated = (num_points_interp <= num_states) and is_interpolated

        # Unpack necessary attributes of the specific period.
        rewards_period = state_space.get_attribute_from_period("rewards", period)
        emaxs_period = state_space.get_attribute_from_period("emaxs", period)[:, :4]
        max_education = (
            state_space.get_attribute_from_period("states", period)[:, 3]
            >= state_space.edu_max
        )

        if any_interpolated:
            # Get indicator for interpolation and simulation of states
            is_simulated = get_simulated_indicator(
                num_points_interp, num_states, period, is_debug
            )

            # Constructing the exogenous variable for all states, including the ones
            # where simulation will take place. All information will be used in either
            # the construction of the prediction model or the prediction step.
            exogenous, max_emax = get_exogenous_variables(
                rewards_period, emaxs_period, shifts, delta, max_education
            )

            # Constructing the dependent variables for all states at the random subset
            # of points where the EMAX is actually calculated.
            endogenous = get_endogenous_variable(
                rewards_period,
                emaxs_period,
                max_emax,
                is_simulated,
                draws_emax_risk,
                delta,
                max_education,
            )

            # Create prediction model based on the random subset of points where the
            # EMAX is actually simulated and thus dependent and independent variables
            # are available. For the interpolation points, the actual values are used.
            emax = get_predictions(endogenous, exogenous, max_emax, is_simulated)

        else:
            emax = construct_emax_risk(
                rewards_period[:, -2:],
                rewards_period[:, :4],
                emaxs_period,
                draws_emax_risk,
                delta,
                max_education,
            )

        state_space.get_attribute_from_period("emaxs", period)[:, 4] = emax

    return state_space


def get_simulated_indicator(num_points_interp, num_states, period, is_debug):
    """Get the indicator for points of interpolation and simulation.

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
        Array of shape (num_states,) indicating states which will be interpolated.

    """
    # Drawing random interpolation points
    interpolation_points = np.random.choice(
        num_states, size=num_points_interp, replace=False
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


def get_exogenous_variables(rewards, emaxs, draws, delta, max_education):
    """Get exogenous variables for interpolation scheme.

    Parameters
    ----------
    rewards : np.ndarray
        Array with shape (num_states_in_period, 9).
    emaxs : np.ndarray
        Array with shape (num_states_in_period, 4).
    draws : np.ndarray
        Array with shape (num_draws, 4).
    delta : float
        Discount factor.
    max_education: np.ndarray
        Array with shape (num_states_in_period,) containing an indicator for whether the
        state has reached maximum education.

    Returns
    -------
    exogenous : np.ndarray
        Array with shape (num_states_in_period, 9).
    max_emax : np.ndarray
        Array with shape (num_states_in_period,) containing maximum emax.

    """
    total_values = get_continuation_value(
        rewards[:, -2:],
        rewards[:, :4],
        emaxs,
        draws.reshape(1, -1),
        delta,
        max_education,
    )

    max_emax = total_values.max(axis=1)
    exogenous = max_emax - total_values.reshape(-1, 4)

    exogenous = np.column_stack(
        (exogenous, np.sqrt(exogenous), np.ones(exogenous.shape[0]))
    )

    return exogenous, max_emax.reshape(-1)


def get_endogenous_variable(
    rewards, emaxs, max_emax, is_simulated, draws_emax_risk, delta, max_education
):
    """Construct endogenous variable for the subset of interpolation points.

    Parameters
    ----------
    rewards : np.ndarray
        Array with shape (num_states_in_period, 9)
    emaxs : np.ndarray
        Array with shape (num_states_in_period, 4).
    max_emax : np.ndarray
        Array with shape (num_states_in_period,) containing maximum of exogenous emax.
    is_simulated : np.ndarray
        Array with shape (num_states_in_period,) containing indicators for simulated
        emaxs.
    draws_emax_risk : np.ndarray
        Array with shape (num_draws, 4) containing draws.
    delta : float
        Discount factor.
    max_education: np.ndarray
        Array with shape (num_states_in_period,) containing an indicator for whether the
        state has reached maximum education.

    """
    emax = construct_emax_risk(
        rewards[:, -2:], rewards[:, :4], emaxs, draws_emax_risk, delta, max_education
    )
    endogenous = emax - max_emax
    endogenous[~is_simulated] = np.nan

    return endogenous


def get_predictions(endogenous, exogenous, maxe, is_simulated):
    """Get ols predictions.

    Fit an OLS regression of the exogenous variables on the endogenous variables and
    use the results to predict the endogenous variables for all points in state space.

    Parameters
    ----------
    endogenous : np.ndarray
        Array with shape (num_simulated_states_in_period,) containing emax for states
        used to interpolate the rest.
    exogenous : np.ndarray
        Array with shape (num_states_in_period, 9) containing exogenous variables.
    maxe : np.ndarray
        Array with shape (num_states_in_period,) containing the maximum emax.
    is_simulated : np.ndarray
        Array with shape (num_states_in_period,) containing indicator for states which
        are used to estimate the coefficients for the interpolation.

    """
    # Define ordinary least squares model and fit to the data.
    beta = ols(endogenous[is_simulated], exogenous[is_simulated])

    # Use the model to predict EMAX for all states. As in Keane & Wolpin (1994),negative
    # predictions are truncated to zero.
    endogenous_predicted = exogenous.dot(beta)
    endogenous_predicted = np.clip(endogenous_predicted, 0.00, None)

    # Construct predicted EMAX for all states and the replace interpolation points with
    # simulated values.
    predictions = endogenous_predicted + maxe
    predictions[is_simulated] = endogenous[is_simulated] + maxe[is_simulated]

    check_prediction_model(endogenous_predicted, beta)

    return predictions


def check_prediction_model(predictions_diff, beta):
    """Perform some basic consistency checks for the prediction model."""
    assert np.all(predictions_diff >= 0.00)
    assert beta.shape == (9,)
    assert np.all(np.isfinite(beta))


def calculate_wages_systematic(states, covariates, coeffs_a, coeffs_b, type_shifts):
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

    if KW_SQUARED_EXPERIENCES:
        exp_a_sq *= 100.00
        exp_b_sq *= 100.00

    relevant_covariates = np.column_stack(
        (
            np.ones(states.shape[0]),
            states[:, [3, 1]],
            exp_a_sq,
            states[:, 2],
            exp_b_sq,
            covariates[:, 9:11],
            states[:, 0],
            covariates[:, 13],
        )
    )

    # Calculate systematic part of wages in OCCUPATION A and OCCUPATION B
    wages = np.full((states.shape[0], 2), np.nan)

    wages[:, 0] = np.column_stack((relevant_covariates, covariates[:, [7, 2]])).dot(
        coeffs_a
    )

    wages[:, 1] = np.column_stack((relevant_covariates, covariates[:, [8, 3]])).dot(
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

    Parameters
    ----------
    a : np.ndarray
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
    >>> res_pandas = pd.get_dummies(a).values
    >>> assert np.allclose(res, res_pandas)

    """
    n_values = np.max(a) + 1
    return np.eye(n_values)[a]


class StateSpace:
    """Class containing all objects related to the state space of a discrete choice
    dynamic programming model.

    Parameters
    ----------
    num_periods : int
        Number of periods.
    num_types : int
        Number of types.
    edu_starts : list
        Contains different initial levels of education for agents.
    edu_max : int
        Maximum level of education for an agent.
    optim_paras : dict
        Contains various information necessary for the calculation of rewards for each
        agent.

    Attributes
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        choice_lagged and type information.
    indexer : np.ndarray
        Array with shape (num_periods, num_periods, num_periods, edu_max, 4, num_types).
    covariates : np.ndarray
        Array with shape (num_periods, 16) containing covariates of each state necessary
        to calculate rewards.
    rewards : np.ndarray
        Array with shape (num_states, 9) containing rewards of each state.
    emaxs : np.ndarray
        Array with shape (num_states, 5) containing containing the emax of each choice
        (OCCUPATION A, OCCUPATION B, SCHOOL, HOME) of the subsequent period and the
        simulated or interpolated maximum of the current period.
    num_periods : int
        Number of periods.
    num_types : int
        Number of types.
    states_columns : list
        List of column names in ``self.states``.
    covariates_columns : list
        List of column names in ``self.covariates``.
    rewards_columns : list
        List of column names in ``self.rewards``.
    emaxs_columns : list
        List of column names in ``self.emaxs``.

    """

    states_columns = ["period", "exp_a", "exp_b", "edu", "choice_lagged", "type"]

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

    emaxs_columns = ["emax_a", "emax_b", "emax_edu", "emax_home", "emax"]

    def __init__(self, num_periods, num_types, edu_starts, edu_max, optim_paras=None):
        self.num_periods = num_periods
        self.num_types = num_types
        self.edu_max = edu_max

        self.states, self.indexer = pyth_create_state_space(
            num_periods, num_types, edu_starts, edu_max
        )
        self.covariates = create_covariates(self.states)

        # Passing :data:`optim_paras` is optional.
        if optim_paras:
            self.rewards = np.column_stack(
                (
                    pyth_calculate_rewards_systematic(
                        self.states, self.covariates, optim_paras
                    )
                )
            )

        self._create_slices_by_periods(num_periods)

    @property
    def states_per_period(self):
        """Get a list of states per period starting from the first period."""
        return np.array([len(range(i.start, i.stop)) for i in self.slices_by_periods])

    @property
    def num_states(self):
        """Get the total number states in the state space."""
        return self.states.shape[0]

    def update_systematic_rewards(self, optim_paras):
        self.rewards = np.column_stack(
            (
                pyth_calculate_rewards_systematic(
                    self.states, self.covariates, optim_paras
                )
            )
        )

    def get_attribute_from_period(self, attr, period):
        """Get an attribute of the state space sliced to a given period.

        Parameters
        ----------
        attr : str
            String of attribute name (e.g. ``"covariates"``).
        period : int
            Attribute is retrieved from this period.

        Raises
        ------
        StateSpaceError
            An error is raised if the attribute or the period is not part of the state
            space.

        """
        try:
            attribute = getattr(self, attr)
        except AttributeError:
            raise StateSpaceError("Inadmissible attribute.")

        try:
            indices = self.slices_by_periods[period]
        except IndexError:
            raise StateSpaceError("Inadmissible period.")

        return attribute[indices]

    def to_frame(self):
        """Get pandas DataFrame of state space.

        Example
        -------
        >>> state_space = StateSpace(1, 1, [10], 11)
        >>> state_space.to_frame().shape
        (2, 22)

        """
        attributes = [
            getattr(self, i, None)
            for i in ["states", "covariates", "rewards", "emaxs"]
            if getattr(self, i, None) is not None
        ]
        columns = [
            item
            for i in ["states", "covariates", "rewards", "emaxs"]
            for item in getattr(self, i + "_columns")
            if getattr(self, i, None) is not None
        ]

        return pd.DataFrame(np.hstack(attributes), columns=columns)

    def _create_slices_by_periods(self, num_periods):
        """Create slices to index all attributes in a given period.

        It is important that the returned objects are not fancy indices. Fancy indexing
        results in copies of array which decrease performance and raise memory usage.

        """
        self.slices_by_periods = []
        for i in range(num_periods):
            idx_start, idx_end = np.where(self.states[:, 0] == i)[0][[0, -1]]
            self.slices_by_periods.append(slice(idx_start, idx_end + 1))
