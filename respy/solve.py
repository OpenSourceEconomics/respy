import numpy as np
import pandas as pd
from numba import guvectorize
from numba import njit

from respy.config import BASE_COVARIATES
from respy.config import HUGE_FLOAT
from respy.config import INADMISSIBILITY_PENALTY
from respy.pre_processing.model_processing import process_options
from respy.pre_processing.model_processing import process_params
from respy.shared import create_base_draws
from respy.shared import transform_disturbances


def solve(params, options):
    """Solve the model.

    This function is a wrapper for the solution routine.

    Parameters
    ----------
    params : pd.DataFrame
        DataFrame containing parameter series.
    options : dict
        Dictionary containing model attributes which are not optimized.

    """
    params, optim_paras = process_params(params)
    options = process_options(options)

    state_space = StateSpace(params, options)
    state_space = solve_with_backward_induction(
        state_space, options["interpolation_points"], optim_paras
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


def create_reward_components(types, covariates, optim_paras):
    """Calculate systematic rewards for each state.

    Wages are only available for some choices, i.e. n_nonpec >= n_wages. We extend the
    array of wages with ones for the difference in dimensions, n_nonpec - n_wages. Ones
    are necessary as it facilitates the aggregation of reward components in
    :func:`calculate_emax_value_functions` and related functions.

    Parameters
    ----------
    types : np.ndarray
        Array with shape (n_states,) containing type information.
    covariates : dict
        Dictionary with covariate arrays for wage and nonpec rewards
    optim_paras : dict
        Contains parameters affected by the optimization.

    """
    wage_labels = ["wage_a", "wage_b"]
    log_wages = np.column_stack(
        [np.dot(covariates[w], optim_paras[w]) for w in wage_labels]
    )

    nonpec_labels = ["nonpec_a", "nonpec_b", "nonpec_edu", "nonpec_home"]
    nonpec = np.column_stack(
        [np.dot(covariates[n], optim_paras[n]) for n in nonpec_labels]
    )

    type_deviations = optim_paras["type_shifts"][types]

    log_wages += type_deviations[:, :2]
    nonpec[:, 2:] += type_deviations[:, 2:]

    wages = np.clip(np.exp(log_wages), 0.0, HUGE_FLOAT)

    # Extend wages to dimension of non-pecuniary rewards.
    additional_dim = nonpec.shape[1] - log_wages.shape[1]
    wages = np.column_stack((wages, np.ones((wages.shape[0], additional_dim))))

    return wages, nonpec


def solve_with_backward_induction(state_space, interpolation_points, optim_paras):
    """Calculate utilities with backward induction.

    Parameters
    ----------
    state_space : class
        State space object.
    interpolation_points : int
        A value of -1 indicates that the interpolation is turned off. If the value is a
        non-zero, positive integer, it indicates the number of states which are used to
        interpolate all the rest.
    optim_paras : dict
        Parameters affected by optimization.

    Returns
    -------
    state_space : class
        State space containing the emax of the subsequent period of each choice, columns
        0-3, as well as the maximum emax of the current period for each state, column 4,
        in ``state_space.emaxs``.

    """
    state_space.emaxs = np.zeros((state_space.num_states, 4))
    state_space.emax = np.zeros(state_space.num_states)

    # For myopic agents, utility of later periods does not play a role.
    if optim_paras["delta"] == 0:
        return state_space

    # Unpack arguments.
    delta = optim_paras["delta"]
    shocks_cholesky = optim_paras["shocks_cholesky"]

    shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)

    for period in reversed(range(state_space.num_periods)):

        if period == state_space.num_periods - 1:
            pass

        else:
            states_period = state_space.get_attribute_from_period("states", period)

            state_space.emaxs = get_continuation_values(
                states_period,
                state_space.indexer,
                state_space.emaxs,
                state_space.emax,
                state_space.edu_max,
            )

        num_states = state_space.states_per_period[period]

        base_draws_sol_period = state_space.base_draws_sol[period]
        draws_emax_risk = transform_disturbances(
            base_draws_sol_period, np.zeros(4), shocks_cholesky
        )

        # Unpack necessary attributes of the specific period.
        wages = state_space.get_attribute_from_period("wages", period)
        nonpec = state_space.get_attribute_from_period("nonpec", period)
        emaxs_period = state_space.get_attribute_from_period("emaxs", period)
        max_education = (
            state_space.get_attribute_from_period("states", period)[:, 3]
            >= state_space.edu_max
        )

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In that case no interpolation is needed.
        any_interpolated = (
            interpolation_points <= num_states and interpolation_points != -1
        )

        if any_interpolated:
            # These shifts are used to determine the expected values of the two labor
            # market alternatives. These are log normal distributed and thus the draws
            # cannot simply set to zero.
            shifts = np.zeros(4)
            shifts[:2] = np.clip(np.exp(np.diag(shocks_cov)[:2] / 2.0), 0.0, HUGE_FLOAT)

            # Get indicator for interpolation and simulation of states. The seed value
            # is the base seed plus the number of the period. Thus, not interpolated
            # states are held constant for each periods and not across periods.
            not_interpolated = get_not_interpolated_indicator(
                interpolation_points, num_states, state_space.seed + period
            )

            # Constructing the exogenous variable for all states, including the ones
            # where simulation will take place. All information will be used in either
            # the construction of the prediction model or the prediction step.
            exogenous, max_emax = calculate_exogenous_variables(
                wages, nonpec, emaxs_period, shifts, delta, max_education
            )

            # Constructing the dependent variables for all states at the random subset
            # of points where the EMAX is actually calculated.
            endogenous = calculate_endogenous_variables(
                wages,
                nonpec,
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
            emax = calculate_emax_value_functions(
                wages, nonpec, emaxs_period, draws_emax_risk, delta, max_education
            )

        state_space.get_attribute_from_period("emax", period)[:] = emax

    return state_space


def get_not_interpolated_indicator(interpolation_points, num_states, seed):
    """Get indicator for states which will be not interpolated.

    Randomness in this function is held constant for each period but not across periods.
    This is done by adding the period to the seed set for the solution.

    Parameters
    ----------
    interpolation_points : int
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
        num_states, size=interpolation_points, replace=False
    )

    # Constructing an indicator whether a state will be interpolated.
    not_interpolated = np.full(num_states, False)
    not_interpolated[interpolation_points] = True

    return not_interpolated


def calculate_exogenous_variables(wages, nonpec, emaxs, draws, delta, max_education):
    """Calculate exogenous variables for interpolation scheme.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (n_states_in_period, n_wages).
    nonpec : np.ndarray
        Array with shape (n_states_in_period, n_choices).
    emaxs : np.ndarray
        Array with shape (n_states_in_period, n_choices).
    draws : np.ndarray
        Array with shape (n_draws, n_choices).
    delta : float
        Discount factor.
    max_education: np.ndarray
        Array with shape (n_states_in_period,) containing an indicator for whether the
        state has reached maximum education.

    Returns
    -------
    exogenous : np.ndarray
        Array with shape (n_states_in_period, n_choices * 2 + 1).
    max_emax : np.ndarray
        Array with shape (n_states_in_period,) containing maximum over all value
        functions.

    """
    value_functions = calculate_value_functions(
        wages, nonpec, emaxs, draws.reshape(1, -1), delta, max_education
    )

    max_value_functions = value_functions.max(axis=1)
    exogenous = max_value_functions - value_functions.reshape(-1, 4)

    exogenous = np.column_stack(
        (exogenous, np.sqrt(exogenous), np.ones(exogenous.shape[0]))
    )

    return exogenous, max_value_functions.reshape(-1)


def calculate_endogenous_variables(
    wages,
    nonpec,
    continuation_values,
    max_value_functions,
    not_interpolated,
    draws,
    delta,
    max_education,
):
    """Calculate endogenous variable for all states which are not interpolated.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (n_states_in_period, n_wages).
    nonpec : np.ndarray
        Array with shape (n_states_in_period, n_choices).
    continuation_values : np.ndarray
        Array with shape (n_states_in_period, n_choices).
    max_value_functions : np.ndarray
        Array with shape (n_states_in_period,) containing maximum over all value
        functions.
    not_interpolated : np.ndarray
        Array with shape (n_states_in_period,) containing indicators for simulated
        continuation_values.
    draws : np.ndarray
        Array with shape (num_draws, n_choices) containing draws.
    delta : float
        Discount factor.
    max_education: np.ndarray
        Array with shape (n_states_in_period,) containing an indicator for whether the
        state has reached maximum education.

    """
    emax_value_functions = calculate_emax_value_functions(
        wages[not_interpolated],
        nonpec[not_interpolated],
        continuation_values[not_interpolated],
        draws,
        delta,
        max_education[not_interpolated],
    )
    endogenous = emax_value_functions - max_value_functions[not_interpolated]

    return endogenous


def get_predictions(endogenous, exogenous, max_value_functions, not_interpolated):
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
        Array with shape (n_states_in_period, n_choices * 2 + 1) containing exogenous
        variables.
    max_value_functions : np.ndarray
        Array with shape (n_states_in_period,) containing the maximum over all value
        functions.
    not_interpolated : np.ndarray
        Array with shape (n_states_in_period,) containing indicator for states which
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
    predictions = endogenous_predicted + max_value_functions
    predictions[not_interpolated] = endogenous + max_value_functions[not_interpolated]

    check_prediction_model(endogenous_predicted, beta)

    return predictions


def check_prediction_model(predictions_diff, beta):
    """Perform some basic consistency checks for the prediction model."""
    assert np.all(predictions_diff >= 0.00)
    assert beta.shape == (9,)
    assert np.all(np.isfinite(beta))


def create_choice_covariates(covariates_df, states_df, params_spec):
    """Create the covariates for each  choice.

    Parameters
    ----------
    covariates_df : pd.DataFrame
        DataFrame with the basic covariates.
    states_df : pd.DataFrame
        DataFrame with the state information.
    params_spec : pd.DataFrame
        The parameter specification.

    Returns
    -------
    wage_covariates: list
        List of length nchoices with covariate arrays for systematic wages.
    nonpec_covariates: list
        List of length nchoices with covariate arrays for nonpecuniary rewards.

    """
    all_data = pd.concat([covariates_df, states_df], axis="columns", sort=False)

    covariates = {}

    for choice in ["a", "b", "edu", "home"]:
        if f"wage_{choice}" in params_spec.index:
            wage_columns = params_spec.loc[f"wage_{choice}"].index
            covariates[f"wage_{choice}"] = all_data[wage_columns].to_numpy()

        nonpec_columns = params_spec.loc[f"nonpec_{choice}"].index
        covariates[f"nonpec_{choice}"] = all_data[nonpec_columns].to_numpy()

    for key, val in covariates.items():
        covariates[key] = np.ascontiguousarray(val)

    return covariates


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
        Array with shape (num_periods, num_periods, num_periods, edu_max, n_choices,
        num_types).
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates of each state necessary
        to calculate rewards.
    wages : np.ndarray
        Array with shape (n_states_in_period, n_choices) which contains zeros in places
        for choices without wages.
    nonpec : np.ndarray
        Array with shape (n_states_in_period, n_choices).
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
    base_covariates_columns : list
        List of column names in ``self.covariates``.
    wages_columns : list
        List of column names in ``self.wages``
    nonpec_columns : list
        List of column names in ``self.nonpec``
    emaxs_columns : list
        List of column names in ``self.emaxs``.

    """

    states_columns = ["period", "exp_a", "exp_b", "edu", "choice_lagged", "type"]

    base_covariates_columns = list(BASE_COVARIATES.keys())

    wages_columns = ["wage_a", "wage_b"]
    nonpec_columns = ["nonpec_a", "nonpec_b", "nonpec_edu", "nonpec_home"]
    emaxs_columns = ["emax_a", "emax_b", "emax_edu", "emax_home", "emax"]

    def __init__(self, params, options):
        params, optim_paras = process_params(params)
        options = process_options(options)

        # Add some arguments to the state space.
        self.edu_max = options["education_max"]
        self.num_periods = options["num_periods"]
        self.num_types = optim_paras["num_types"]
        self.seed = options["solution_seed"]

        self.base_draws_sol = create_base_draws(
            (self.num_periods, options["solution_draws"], 4), self.seed
        )

        self.states, self.indexer = create_state_space(
            self.num_periods, self.num_types, options["education_start"], self.edu_max
        )
        states_df = pd.DataFrame(data=self.states, columns=self.states_columns)

        base_covariates_df = create_base_covariates(states_df, options["covariates"])

        self.covariates = create_choice_covariates(
            base_covariates_df, states_df, params
        )

        self.wages, self.nonpec = create_reward_components(
            self.states[:, 5], self.covariates, optim_paras
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
        self.wages, self.nonpec = create_reward_components(
            self.states[:, 5], self.covariates, optim_paras
        )

    def get_attribute_from_period(self, attr, period):
        """Get an attribute of the state space sliced to a given period.

        Parameters
        ----------
        attr : str
            String of attribute name, e.g. ``"states"``.
        period : int
            Attribute is retrieved from this period.

        """
        if attr == "covariates":
            raise AttributeError("Attribute covariates cannot be retrieved by periods.")
        else:
            pass

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
def calculate_emax_value_functions(
    wages,
    nonpec,
    continuation_values,
    draws,
    delta,
    max_education,
    emax_value_functions,
):
    r"""Calculate the expected maximum of value functions for a set of unobservables.

    The function takes an agent and calculates the utility for each of the choices, the
    ex-post rewards, with multiple draws from the distribution of unobservables and adds
    the discounted expected maximum utility of subsequent periods resulting from
    choices. Averaging over all maximum utilities yields the expected maximum utility of
    this state.

    The underlying process in this function is called `Monte Carlo integration`_. The
    goal is to approximate an integral by evaluating the integrand at randomly chosen
    points. In this setting, one wants to approximate the expected maximum utility of
    the current state.

    Note that ``wages`` have the same length as ``nonpec`` despite that wages are only
    available in some sectors. Missing sectors are filled with ones. In the case of a
    sector with wage and without wage, flow utilities are

    .. math::

        \text{Flow Utility} = \text{Wage} * \epsilon + \text{Non-pecuniary}
        \text{Flow Utility} = 1 * \epsilon + \text{Non-pecuniary}


    Parameters
    ----------
    wages : np.ndarray
        Array with shape (n_choices,) containing wages.
    nonpec : np.ndarray
        Array with shape (n_choices,) containing non-pecuniary rewards.
    continuation_values : np.ndarray
        Array with shape (n_choices,) containing expected maximum utility for each
        choice in the subsequent period.
    draws : np.ndarray
        Array with shape (num_draws, n_choices).
    delta : float
        The discount factor.
    max_education: bool
        Indicator for whether the state has reached maximum education.

    Returns
    -------
    emax_value_functions : float
        Expected maximum utility of an agent.

    .. _Monte Carlo integration:
        https://en.wikipedia.org/wiki/Monte_Carlo_integration

    """
    num_draws, num_choices = draws.shape

    emax_value_functions[0] = 0.0

    for i in range(num_draws):

        max_value_functions = 0.0

        for j in range(num_choices):
            flow_utility = wages[j] * draws[i, j] + nonpec[j]
            value_function = flow_utility + delta * continuation_values[j]

            if j == 2 and max_education:
                value_function += INADMISSIBILITY_PENALTY

            if value_function > max_value_functions:
                max_value_functions = value_function

        emax_value_functions[0] += max_value_functions

    emax_value_functions[0] /= num_draws


@njit(nogil=True)
def get_continuation_values(
    states, indexer, continuation_values, emax_value_functions, edu_max
):
    """Get the maximum utility from the subsequent period.

    This function takes a parent state and looks up the continuation value for each
    of the four choices in the following period.

    """
    for i in range(states.shape[0]):
        # Unpack parent state and get index.
        period, exp_a, exp_b, edu, choice_lagged, type_ = states[i]
        k_parent = indexer[period, exp_a, exp_b, edu, choice_lagged - 1, type_]

        # Working in Occupation A in period + 1
        k = indexer[period + 1, exp_a + 1, exp_b, edu, 0, type_]
        continuation_values[k_parent, 0] = emax_value_functions[k]

        # Working in Occupation B in period +1
        k = indexer[period + 1, exp_a, exp_b + 1, edu, 1, type_]
        continuation_values[k_parent, 1] = emax_value_functions[k]

        # Schooling in period + 1. Note that adding an additional year of schooling is
        # only possible for those that have strictly less than the maximum level of
        # additional education allowed. This condition is necessary as there are states
        # which have reached maximum education. Incrementing education by one would
        # target an inadmissible state.
        if edu >= edu_max:
            continuation_values[k_parent, 2] = 0.0
        else:
            k = indexer[period + 1, exp_a, exp_b, edu + 1, 2, type_]
            continuation_values[k_parent, 2] = emax_value_functions[k]

        # Staying at home in period + 1
        k = indexer[period + 1, exp_a, exp_b, edu, 3, type_]
        continuation_values[k_parent, 3] = emax_value_functions[k]

    return continuation_values


@guvectorize(
    [
        "f4[:], f4[:], f4[:], f4[:, :], f4, b1, f4[:, :]",
        "f8[:], f8[:], f8[:], f8[:, :], f8, b1, f8[:, :]",
    ],
    "(m), (n), (n), (p, n), (), () -> (n, p)",
    nopython=True,
    target="cpu",
)
def calculate_value_functions(
    wages, nonpec, continuation_values, draws, delta, max_education, value_functions
):
    """Calculate choice-specific value functions.

    This function is a reduced version of
    :func:`calculate_value_functions_and_flow_utilities` which does not return flow
    utilities. The reason is that a second return argument doubles runtime whereas it is
    only needed during simulation.

    """
    num_draws, num_choices = draws.shape

    for i in range(num_draws):
        for j in range(num_choices):
            flow_utility = wages[j] * draws[i, j] + nonpec[j]
            value_function = flow_utility + delta * continuation_values[j]

            if j == 2 and max_education:
                value_function += INADMISSIBILITY_PENALTY

            value_functions[j, i] = value_function


def create_base_covariates(states, covariates_spec):
    """Create set of covariates for each state.

    Parameters
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        choice_lagged and type of each state.
    covariates_spec : dict
        Dictionary where keys represent covariates and values are strings which can be
        passed to pd.eval.

    Returns
    -------
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates of each state.

    """
    covariates = states.copy()

    for covariate, definition in covariates_spec.items():
        covariates[covariate] = covariates.eval(definition)

    covariates = covariates.drop(columns=states.columns).astype(float)

    return covariates


def ols(y, x):
    """Calculate OLS coefficients using a pseudo-inverse.

    Parameters
    ----------
    x : np.ndarray
        n x n matrix of independent variables.
    y : np.ndarray
        n x 1 matrix with dependent variable.

    Returns
    -------
    beta : np.ndarray
        n x 1 array of estimated parameter vector

    """
    beta = np.dot(np.linalg.pinv(x.T.dot(x)), x.T.dot(y))
    return beta


def mse(x1, x2, axis=0):
    """Calculate mean squared error.

    If ``x1`` and ``x2`` have different shapes, then they need to broadcast. This uses
    ``np.asanyarray`` to convert the input. Whether this is the desired result or not
    depends on the array subclass, for example NumPy matrices will silently produce an
    incorrect result.

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two arrays.
    axis : int
       Axis along which the summary statistic is calculated

    Returns
    -------
    mse : ndarray or float
       Mean squared error along given axis.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1 - x2) ** 2, axis=axis)


def rmse(x1, x2, axis=0):
    """Calculate root mean squared error.

    If ``x1`` and ``x2`` have different shapes, then they need to broadcast. This uses
    ``np.asanyarray`` to convert the input. Whether this is the desired result or not
    depends on the array subclass, for example NumPy matrices will silently produce an
    incorrect result.

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    rmse : np.ndarray or float
       root mean squared error along given axis.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))
