import numpy as np
import pandas as pd
from numba import guvectorize
from numba import njit

from respy.config import HUGE_FLOAT
from respy.config import INADMISSIBILITY_PENALTY
from respy.pre_processing.model_processing import process_model_spec
from respy.shared import create_base_draws
from respy.shared import transform_disturbances


def solve(params_spec, options_spec):
    """Solve the model.

    This function is a wrapper for the solution routine.

    Parameters
    ----------
    params_spec : pd.DataFrame
        DataFrame containing parameter series.
    options_spec : dict
        Dictionary containing model attributes which are not optimized.

    """
    attr, optim_paras = process_model_spec(params_spec, options_spec)

    state_space = StateSpace(attr, optim_paras)

    state_space = solve_with_backward_induction(
        state_space, attr["interpolation"], attr["num_points_interp"], optim_paras
    )

    return state_space


@njit
def create_state_space(num_periods, num_types, edu_starts, edu_max):
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
    >>> states, indexer = create_state_space(
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


def create_systematic_rewards(states, covariates, optim_paras):
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


def solve_with_backward_induction(
    state_space, interpolation, num_points_interp, optim_paras
):
    """Calculate utilities with backward induction.

    Parameters
    ----------
    state_space : class
        State space object.
    interpolation : np.array
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

        base_draws_sol_period = state_space.base_draws_sol[period]
        draws_emax_risk = transform_disturbances(
            base_draws_sol_period, np.zeros(4), shocks_cholesky
        )

        # Unpack necessary attributes of the specific period.
        rewards_period = state_space.get_attribute_from_period("rewards", period)
        emaxs_period = state_space.get_attribute_from_period("emaxs", period)[:, :4]
        max_education = (
            state_space.get_attribute_from_period("states", period)[:, 3]
            >= state_space.edu_max
        )

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In that case no interpolation is needed.
        any_interpolated = (num_points_interp <= num_states) and interpolation

        if any_interpolated:
            # Get indicator for interpolation and simulation of states. The seed value
            # is the base seed plus the number of the period. Thus, not interpolated
            # states are held constant for each periods and not across periods.
            not_interpolated = get_not_interpolated_indicator(
                num_points_interp, num_states, state_space.seed + period
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
                not_interpolated,
                draws_emax_risk,
                delta,
                max_education,
            )

            # Create prediction model based on the random subset of points where the
            # EMAX is actually simulated and thus dependent and independent variables
            # are available. For the interpolation points, the actual values are used.
            emax = get_predictions(endogenous, exogenous, max_emax, not_interpolated)

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


def get_not_interpolated_indicator(num_points_interp, num_states, seed):
    """Get indicator for states which will be not interpolated.

    Randomness in this function is held constant for each period but not across periods.
    This is done by adding the period to the seed set for the solution.

    Parameters
    ----------
    num_points_interp : int
        Number of states which will be interpolated.
    num_states : int
        Total number of states in period.
    seed : int
        Seed to set randomness.

    Returns
    -------
    not_interpolated : np.ndarray
        Array of shape (num_states,) indicating states which will not be interpolated.

    """
    np.random.seed(seed)

    # Drawing random interpolation indices.
    interpolation_points = np.random.choice(
        num_states, size=num_points_interp, replace=False
    )

    # Constructing an indicator whether a state will be interpolated.
    not_interpolated = np.full(num_states, False)
    not_interpolated[interpolation_points] = True

    return not_interpolated


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
    rewards, emaxs, max_emax, not_interpolated, draws_emax_risk, delta, max_education
):
    """Construct endogenous variable for all states which are not interpolated.

    Parameters
    ----------
    rewards : np.ndarray
        Array with shape (num_states_in_period, 9)
    emaxs : np.ndarray
        Array with shape (num_states_in_period, 4).
    max_emax : np.ndarray
        Array with shape (num_states_in_period,) containing maximum of exogenous emax.
    not_interpolated : np.ndarray
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
        rewards[:, -2:][not_interpolated],
        rewards[:, :4][not_interpolated],
        emaxs[not_interpolated],
        draws_emax_risk,
        delta,
        max_education[not_interpolated],
    )
    endogenous = emax - max_emax[not_interpolated]

    return endogenous


def get_predictions(endogenous, exogenous, maxe, not_interpolated):
    """Get predictions for the emax of interpolated states.

    Fit an OLS regression of the exogenous variables on the endogenous variables and use
    the results to predict the endogenous variables for all points in state space. Then,
    replace emax values for not interpolated states with true value.

    Parameters
    ----------
    endogenous : np.ndarray
        Array with shape (num_simulated_states_in_period,) containing emax for states
        used to interpolate the rest.
    exogenous : np.ndarray
        Array with shape (num_states_in_period, 9) containing exogenous variables.
    maxe : np.ndarray
        Array with shape (num_states_in_period,) containing the maximum emax.
    not_interpolated : np.ndarray
        Array with shape (num_states_in_period,) containing indicator for states which
        are not interpolated and used to estimate the coefficients for the
        interpolation.

    """
    # Define ordinary least squares model and fit to the data.
    beta = ols(endogenous, exogenous[not_interpolated])

    # Use the model to predict EMAX for all states. As in Keane & Wolpin (1994),
    # negative predictions are truncated to zero.
    endogenous_predicted = exogenous.dot(beta)
    endogenous_predicted = np.clip(endogenous_predicted, 0.00, None)

    # Construct predicted EMAX for all states and the
    predictions = endogenous_predicted + maxe
    predictions[not_interpolated] = endogenous + maxe[not_interpolated]

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

    """
    exp_a_sq = states[:, 1] ** 2 / 100
    exp_b_sq = states[:, 2] ** 2 / 100

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


def create_choice_covariates(covariates_df, states_df, params_spec):
    """Create the covariates for each  choice.

    Parameters
    ----------
    covariates_df: DataFrame
        DataFrame with the basic covariates
    states_df: DataFrame
        DataFrame with the state information
    params_spec: DataFrame
        The parameter specification

    Returns
    -------
    wage_covariates: list
        List of length nchoices with covariate arrays for systematic wages
    nonpec_covariates: list
        List of length nchoices with covariate arrays for nonpecuniary rewards

    """
    all_data = pd.concat([covariates_df, states_df], axis=1)
    all_data["constant"] = 1
    all_data["exp_a_sq"] = all_data["exp_a"] ** 2 / 100
    all_data["exp_b_sq"] = all_data["exp_b"] ** 2 / 100

    wage_covariates = []
    nonpec_covariates = []

    for choice in ["a", "b", "edu", "home"]:
        if f"wage_{choice}" in params_spec.index:
            wage_columns = params_spec.loc[f"wage_{choice}"].index
            wage_covariates.append(all_data[wage_columns].to_numpy())
        else:
            wage_covariates.append(None)

        nonpec_columns = params_spec.loc[f"nonpec_{choice}"].index
        nonpec_covariates.append(all_data[nonpec_columns].to_numpy())

    return wage_covariates, nonpec_covariates


class StateSpace:
    """Class containing all objects related to the state space of a discrete choice
    dynamic programming model.

    Parameters
    ----------
    attr : dict
        Dictionary containing model attributes.
    optim_paras : dict
        Dictionary containing parameters affected by optimization.

    Attributes
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        choice_lagged and type information.
    indexer : np.ndarray
        Array with shape (num_periods, num_periods, num_periods, edu_max, 4, num_types).
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates of each state necessary
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

    def __init__(self, attr, optim_paras):
        # Add some arguments to the state space.
        self.edu_max = attr["edu_spec"]["max"]
        self.num_periods = attr["num_periods"]
        self.num_types = optim_paras["num_types"]
        self.seed = attr["seed_sol"]

        self.base_draws_sol = create_base_draws(
            (self.num_periods, attr["num_draws_sol"], 4), self.seed
        )

        self.states, self.indexer = create_state_space(
            self.num_periods, self.num_types, attr["edu_spec"]["start"], self.edu_max
        )
        self.covariates = create_base_covariates(self.states)

        self.states_df = pd.DataFrame(data=self.states, columns=self.states_columns)

        self.covariates_df = pd.DataFrame(
            data=self.covariates, columns=self.covariates_columns
        )

        self.rewards = np.column_stack(
            (create_systematic_rewards(self.states, self.covariates, optim_paras))
        )

        self._create_slices_by_periods(self.num_periods)

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
            (create_systematic_rewards(self.states, self.covariates, optim_paras))
        )

    def get_attribute_from_period(self, attr, period):
        """Get an attribute of the state space sliced to a given period.

        Parameters
        ----------
        attr : str
            String of attribute name (e.g. ``"covariates"``).
        period : int
            Attribute is retrieved from this period.

        """
        try:
            attribute = getattr(self, attr)
        except AttributeError as e:
            raise AttributeError(f"StateSpace has no attribute {attr}.").with_traceback(
                e.__traceback__
            )

        try:
            indices = self.slices_by_periods[period]
        except IndexError as e:
            raise IndexError(f"StateSpace has no period {period}.").with_traceback(
                e.__traceback__
            )

        return attribute[indices]

    def to_frame(self):
        """Get pandas DataFrame of state space.

        Example
        -------
        >>> params_spec, options_spec = generate_random_model()
        >>> attr, optim_paras = process_model_spec(params_spec, options_spec)
        >>> state_space = StateSpace(attr, optim_paras)
        >>> state_space.to_frame().shape
        (18, 31)

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


@guvectorize(
    [
        "f4[:], f4[:], f4[:], f4[:, :], f4, b1, f4[:]",
        "f8[:], f8[:], f8[:], f8[:, :], f8, b1, f8[:]",
    ],
    "(m), (n), (n), (p, n), (), () -> ()",
    nopython=True,
    target="parallel",
)
def construct_emax_risk(
    wages, rewards_systematic, emaxs, draws, delta, max_education, cont_value
):
    """Simulate expected maximum utility for a given distribution of the unobservables.

    The function takes an agent and calculates the utility for each of the choices, the
    ex-post rewards, with multiple draws from the distribution of unobservables and adds
    the discounted expected maximum utility of subsequent periods resulting from
    choices. Averaging over all maximum utilities yields the expected maximum utility of
    this state.

    The underlying process in this function is called `Monte Carlo integration`_. The
    goal is to approximate an integral by evaluating the integrand at randomly chosen
    points. In this setting, one wants to approximate the expected maximum utility of
    the current state.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (2,) containing wages.
    rewards_systematic : np.ndarray
        Array with shape (4,) containing systematic rewards.
    emaxs : np.ndarray
        Array with shape (4,) containing expected maximum utility for each choice in the
        subsequent period.
    draws : np.ndarray
        Array with shape (num_draws, 4).
    delta : float
        The discount factor.
    max_education: bool
        Indicator for whether the state has reached maximum education.

    Returns
    -------
    cont_value : float
        Expected maximum utility of an agent.

    .. _Monte Carlo integration:
        https://en.wikipedia.org/wiki/Monte_Carlo_integration

    """
    num_draws, num_choices = draws.shape
    num_wages = wages.shape[0]

    cont_value[0] = 0.0

    for i in range(num_draws):

        current_max_emax = 0.0

        for j in range(num_choices):
            if j < num_wages:
                rew_ex = wages[j] * draws[i, j] + rewards_systematic[j] - wages[j]
            else:
                rew_ex = rewards_systematic[j] + draws[i, j]

            emax_choice = rew_ex + delta * emaxs[j]

            if j == 2 and max_education:
                emax_choice += INADMISSIBILITY_PENALTY

            if emax_choice > current_max_emax:
                current_max_emax = emax_choice

        cont_value[0] += current_max_emax

    cont_value[0] /= num_draws


@njit(nogil=True)
def get_emaxs_of_subsequent_period(states, indexer, emaxs, edu_max):
    """Get the maximum utility from the subsequent period.

    This function takes a parent node and looks up the utility from each of the four
    choices in the subsequent period.

    Warning
    -------
    This function must be extremely performant as the lookup is done for each state in a
    state space (except for states in the last period) for each evaluation of the
    optimization of parameters.

    """
    for i in range(states.shape[0]):
        # Unpack parent state and get index.
        period, exp_a, exp_b, edu, choice_lagged, type_ = states[i]
        k_parent = indexer[period, exp_a, exp_b, edu, choice_lagged - 1, type_]

        # Working in Occupation A in period + 1
        k = indexer[period + 1, exp_a + 1, exp_b, edu, 0, type_]
        emaxs[k_parent, 0] = emaxs[k, 4]

        # Working in Occupation B in period +1
        k = indexer[period + 1, exp_a, exp_b + 1, edu, 1, type_]
        emaxs[k_parent, 1] = emaxs[k, 4]

        # Schooling in period + 1. Note that adding an additional year of schooling is
        # only possible for those that have strictly less than the maximum level of
        # additional education allowed. This condition is necessary as there are states
        # which have reached maximum education. Incrementing education by one would
        # target an inadmissible state.
        if edu >= edu_max:
            emaxs[k_parent, 2] = 0.0
        else:
            k = indexer[period + 1, exp_a, exp_b, edu + 1, 2, type_]
            emaxs[k_parent, 2] = emaxs[k, 4]

        # Staying at home in period + 1
        k = indexer[period + 1, exp_a, exp_b, edu, 3, type_]
        emaxs[k_parent, 3] = emaxs[k, 4]

    return emaxs


@guvectorize(
    [
        "f4[:], f4[:], f4[:], f4[:, :], f4, b1, f4[:, :]",
        "f8[:], f8[:], f8[:], f8[:, :], f8, b1, f8[:, :]",
    ],
    "(m), (n), (n), (p, n), (), () -> (n, p)",
    nopython=True,
    target="cpu",
)
def get_continuation_value(
    wages, rewards_systematic, emaxs, draws, delta, max_education, cont_value
):
    """Calculate the continuation value.

    This function is a reduced version of
    ``get_continutation_value_and_ex_post_rewards`` which does not return ex post
    rewards. The reason is that a second return argument doubles runtime whereas it is
    only needed during simulation.

    """
    num_draws, num_choices = draws.shape
    num_wages = wages.shape[0]

    for i in range(num_draws):
        for j in range(num_choices):
            if j < num_wages:
                rew_ex = wages[j] * draws[i, j] + rewards_systematic[j] - wages[j]
            else:
                rew_ex = rewards_systematic[j] + draws[i, j]

            cont_value_ = rew_ex + delta * emaxs[j]

            if j == 2 and max_education:
                cont_value_ += INADMISSIBILITY_PENALTY

            cont_value[j, i] = cont_value_


def create_base_covariates(states):
    """Create set of covariates for each state.

    Parameters
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        choice_lagged and type of each state.

    Returns
    -------
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates of each state.

    Examples
    --------
    This example is to benchmark alternative implementations, but even this version does
    not benefit from Numba anymore.

    >>> states, _ = create_state_space(40, 5, [10], 20)
    >>> covariates = create_base_covariates(states)
    >>> assert covariates.shape == (states.shape[0], 16)

    """
    covariates = np.zeros((states.shape[0], 16), dtype=np.int8)

    # Experience in A or B, but not in the last period.
    covariates[:, 0] = np.where((states[:, 1] > 0) & (states[:, 4] != 1), 1, 0)
    covariates[:, 1] = np.where((states[:, 2] > 0) & (states[:, 4] != 2), 1, 0)

    # Last occupation was A, B, or education.
    covariates[:, 2] = np.where(states[:, 4] == 1, 1, 0)
    covariates[:, 3] = np.where(states[:, 4] == 2, 1, 0)
    covariates[:, 4] = np.where(states[:, 4] == 3, 1, 0)

    # No experience in A or B.
    covariates[:, 5] = np.where(states[:, 1] == 0, 1, 0)
    covariates[:, 6] = np.where(states[:, 2] == 0, 1, 0)

    # Any experience in A or B.
    covariates[:, 7] = np.where(states[:, 1] > 0, 1, 0)
    covariates[:, 8] = np.where(states[:, 2] > 0, 1, 0)

    # High school or college graduate
    covariates[:, 9] = np.where(states[:, 3] >= 12, 1, 0)
    covariates[:, 10] = np.where(states[:, 3] >= 16, 1, 0)

    # Was not in school last period and is/is not high school graduate
    covariates[:, 11] = np.where(
        (covariates[:, 4] == 0) & (covariates[:, 9] == 0), 1, 0
    )
    covariates[:, 12] = np.where(
        (covariates[:, 4] == 0) & (covariates[:, 9] == 1), 1, 0
    )

    # Define age groups minor (period < 2), young adult (2 <= period <= 4) and adult (5
    # <= period).
    covariates[:, 13] = np.where(states[:, 0] < 2, 1, 0)
    covariates[:, 14] = np.where(np.isin(states[:, 0], [2, 3, 4]), 1, 0)
    covariates[:, 15] = np.where(states[:, 0] >= 5, 1, 0)

    return covariates


def calculate_rewards_general(covariates, coeffs_a, coeffs_b):
    """Calculate general rewards.

    Parameters
    ----------
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates.
    coeffs_a : np.ndarray
        Array with shape (3,) containing coefficients.
    coeffs_b : np.ndarray
        Array with shape (3,) containing coefficients.

    Returns
    -------
    rewards_general : np.ndarray
        Array with shape (num_states, 2) containing general rewards of occupation.

    Example
    -------
    >>> params_spec, options_spec = generate_random_model()
    >>> attr, optim_paras = process_model_spec(params_spec, options_spec)
    >>> state_space = StateSpace(attr, optim_paras)
    >>> coeffs_a, coeffs_b = np.array([0.05, 0.6, 0.4]), np.array([0.36, 0.7, 1])
    >>> calculate_rewards_general(
    ...     state_space.covariates, coeffs_a, coeffs_b
    ... ).reshape(-1)
    array([0.45, 1.36, 0.45, 1.36, 0.45, 1.36, 0.45, 1.36, 0.45, 1.36, 0.45,
           1.36, 0.45, 1.36, 0.45, 1.36, 0.45, 0.36, 0.05, 1.36, 0.45, 1.36,
           0.45, 1.36, 0.45, 0.36, 0.05, 1.36, 0.45, 1.36, 0.45, 1.36, 0.45,
           0.36, 0.05, 1.36])

    """
    num_states = covariates.shape[0]
    rewards_general = np.full((num_states, 2), np.nan)

    rewards_general[:, 0] = np.column_stack(
        (np.ones(num_states), covariates[:, [0, 5]])
    ).dot(coeffs_a)
    rewards_general[:, 1] = np.column_stack(
        (np.ones(num_states), covariates[:, [1, 6]])
    ).dot(coeffs_b)

    return rewards_general


def calculate_rewards_common(covariates, coeffs_common):
    """Calculate common rewards.

    Covariates 9 and 10 are indicators for high school and college graduates.

    Parameters
    ----------
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates.
    coeffs_common : np.ndarray
        Array with shape (2,) containing coefficients for high school and college
        graduates.

    Returns
    -------
    np.ndarray
        Array with shape (num_states, 1) containing common rewards. Reshaping is
        necessary to broadcast the array over rewards with shape (num_states, 4).

    Example
    -------
    >>> params_spec, options_spec = generate_random_model()
    >>> attr, optim_paras = process_model_spec(params_spec, options_spec)
    >>> state_space = StateSpace(attr, optim_paras)
    >>> coeffs_common = np.array([0.05, 0.6])
    >>> calculate_rewards_common(state_space.covariates, coeffs_common).reshape(-1)
    array([0.  , 0.  , 0.05, 0.05, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.05,
           0.05, 0.05, 0.05, 0.  , 0.  , 0.  , 0.  ])

    """
    return covariates[:, 9:11].dot(coeffs_common).reshape(-1, 1)


def ols(y, x):
    """Ols implementation using a pseudo inverse.

    Parameters
    ----------
    x (ndarray): n x n matrix of independent variables.
    y (array): n x 1 matrix with dependant variable.

    Returns
    -------
        beta (array): n x 1 array of estimated parameter vector

    """
    beta = np.dot(np.linalg.pinv(x.T.dot(x)), x.T.dot(y))
    return beta


def mse(x1, x2, axis=0):
    """mean squared error.

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1 - x2) ** 2, axis=axis)


def rmse(x1, x2, axis=0):
    """root mean squared error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    rmse : ndarray or float
       root mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))
