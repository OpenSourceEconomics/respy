"""Everything related to the solution of a structural model."""
import warnings

import numba as nb
import numpy as np

from respy.config import HUGE_FLOAT
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import aggregate_keane_wolpin_utility
from respy.shared import calculate_value_functions_and_flow_utilities
from respy.shared import clip
from respy.shared import transform_base_draws_with_cholesky_factor
from respy.state_space import StateSpace


def solve(params, options):
    """Solve the model.

    This function takes a model specification and returns the state space of the model
    along with components of the solution such as covariates, non-pecuniary rewards,
    wages, continuation values and value functions as attributes of the class.

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame containing parameter series.
    options : dict
        Dictionary containing model attributes which are not optimized.

    Returns
    -------
    state_space : :class:`~respy.state_space.StateSpace`
        State space of the model which is already solved via backward-induction.

    """
    optim_paras, options = process_params_and_options(params, options)

    state_space = StateSpace(optim_paras, options)
    state_space = solve_with_backward_induction(state_space, optim_paras, options)

    return state_space


def solve_with_backward_induction(state_space, optim_paras, options):
    """Calculate utilities with backward induction.

    Parameters
    ----------
    state_space : :class:`~respy.state_space.StateSpace`
        State space of the model which is not solved yet.
    optim_paras : dict
        Parsed model parameters affected by the optimization.
    options : dict
        Optimization independent model options.

    Returns
    -------
    state_space : :class:`~respy.state_space.StateSpace`

    """
    n_choices = len(optim_paras["choices"])
    n_wages = len(optim_paras["choices_w_wage"])
    n_periods = optim_paras["n_periods"]
    n_states = state_space.states.shape[0]

    state_space.emax_value_functions = np.zeros(n_states)

    # For myopic agents, utility of later periods does not play a role.
    if optim_paras["delta"] == 0:
        return state_space

    # Unpack arguments.
    delta = optim_paras["delta"]
    shocks_cholesky = optim_paras["shocks_cholesky"]

    shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)

    draws_emax_risk = transform_base_draws_with_cholesky_factor(
        state_space.base_draws_sol, shocks_cholesky, n_wages
    )

    for period in reversed(range(n_periods)):
        # Unpack necessary attributes of the specific period.
        wages = state_space.get_attribute_from_period("wages", period)
        nonpec = state_space.get_attribute_from_period("nonpec", period)
        is_inadmissible = state_space.get_attribute_from_period(
            "is_inadmissible", period
        )
        continuation_values = state_space.get_continuation_values(period)

        n_states_in_period = wages.shape[0]

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In that case no interpolation is needed.
        any_interpolated = (
            options["interpolation_points"] <= n_states_in_period
            and options["interpolation_points"] != -1
        )

        if any_interpolated:
            # These shifts are used to determine the expected values of the working
            # alternatives. These are log normal distributed and thus the draws cannot
            # simply set to zero, but :math:`E(X) = \exp\{\mu + \frac{\sigma^2}{2}\}`.
            shifts = np.zeros(n_choices)
            n_choices_w_wage = len(optim_paras["choices_w_wage"])
            shifts[:n_choices_w_wage] = np.clip(
                np.exp(np.diag(shocks_cov)[:n_choices_w_wage] / 2.0), 0.0, HUGE_FLOAT
            )

            # Get indicator for interpolation and simulation of states. The seed value
            # is the base seed plus the number of the period. Thus, not interpolated
            # states are held constant for each periods and not across periods.
            not_interpolated = get_not_interpolated_indicator(
                options["interpolation_points"],
                n_states_in_period,
                next(options["solution_seed_iteration"]),
            )

            # Constructing the exogenous variable for all states, including the ones
            # where simulation will take place. All information will be used in either
            # the construction of the prediction model or the prediction step.
            exogenous, max_emax = calculate_exogenous_variables(
                wages, nonpec, continuation_values, shifts, delta, is_inadmissible
            )

            # Constructing the dependent variables for all states at the random subset
            # of points where the EMAX is actually calculated.
            endogenous = calculate_endogenous_variables(
                wages,
                nonpec,
                continuation_values,
                max_emax,
                not_interpolated,
                draws_emax_risk[period],
                delta,
                is_inadmissible,
            )

            # Create prediction model based on the random subset of points where the
            # EMAX is actually simulated and thus dependent and independent variables
            # are available. For the interpolation points, the actual values are used.
            emax = get_predictions(endogenous, exogenous, max_emax, not_interpolated)

        else:
            emax = calculate_emax_value_functions(
                wages,
                nonpec,
                continuation_values,
                draws_emax_risk[period],
                delta,
                is_inadmissible,
            )

        state_space.get_attribute_from_period("emax_value_functions", period)[:] = emax

    return state_space


def get_not_interpolated_indicator(interpolation_points, n_states, seed):
    """Get indicator for states which will be not interpolated.

    Randomness in this function is held constant for each period but not across periods.
    This is done by adding the period to the seed set for the solution.

    Parameters
    ----------
    interpolation_points : int
        Number of states which will be interpolated.
    n_states : int
        Total number of states in period.
    seed : int
        Seed to set randomness.

    Returns
    -------
    not_interpolated : numpy.ndarray
        Array of shape (n_states,) indicating states which will not be interpolated.

    """
    np.random.seed(seed)

    indices = np.random.choice(n_states, size=interpolation_points, replace=False)

    not_interpolated = np.full(n_states, False)
    not_interpolated[indices] = True

    return not_interpolated


def calculate_exogenous_variables(wages, nonpec, emaxs, draws, delta, is_inadmissible):
    """Calculate exogenous variables for interpolation scheme.

    Parameters
    ----------
    wages : numpy.ndarray
        Array with shape (n_states_in_period, n_wages).
    nonpec : numpy.ndarray
        Array with shape (n_states_in_period, n_choices).
    emaxs : numpy.ndarray
        Array with shape (n_states_in_period, n_choices).
    draws : numpy.ndarray
        Array with shape (n_draws, n_choices).
    delta : float
        Discount factor.
    is_inadmissible : numpy.ndarray
        Array with shape (n_states_in_period,) containing an indicator for whether the
        state has reached maximum education.

    Returns
    -------
    exogenous : numpy.ndarray
        Array with shape (n_states_in_period, n_choices * 2 + 1).
    max_emax : numpy.ndarray
        Array with shape (n_states_in_period,) containing maximum over all value
        functions.

    """
    value_functions, _ = calculate_value_functions_and_flow_utilities(
        wages, nonpec, emaxs, draws, delta, is_inadmissible
    )

    max_value_functions = value_functions.max(axis=1)
    exogenous = max_value_functions.reshape(-1, 1) - value_functions

    exogenous = np.column_stack(
        (exogenous, np.sqrt(exogenous), np.ones(exogenous.shape[0]))
    )

    return exogenous, max_value_functions


def calculate_endogenous_variables(
    wages,
    nonpec,
    continuation_values,
    max_value_functions,
    not_interpolated,
    draws,
    delta,
    is_inadmissible,
):
    """Calculate endogenous variable for all states which are not interpolated.

    Parameters
    ----------
    wages : numpy.ndarray
        Array with shape (n_states_in_period, n_wages).
    nonpec : numpy.ndarray
        Array with shape (n_states_in_period, n_choices).
    continuation_values : numpy.ndarray
        Array with shape (n_states_in_period, n_choices).
    max_value_functions : numpy.ndarray
        Array with shape (n_states_in_period,) containing maximum over all value
        functions.
    not_interpolated : numpy.ndarray
        Array with shape (n_states_in_period,) containing indicators for simulated
        continuation_values.
    draws : numpy.ndarray
        Array with shape (n_draws, n_choices) containing draws.
    delta : float
        Discount factor.
    is_inadmissible : numpy.ndarray
        Array with shape (n_states_in_period,) containing an indicator for whether the
        state has reached maximum education.

    """
    emax_value_functions = calculate_emax_value_functions(
        wages[not_interpolated],
        nonpec[not_interpolated],
        continuation_values[not_interpolated],
        draws,
        delta,
        is_inadmissible[not_interpolated],
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
    endogenous : numpy.ndarray
        Array with shape (num_simulated_states_in_period,) containing emax for states
        used to interpolate the rest.
    exogenous : numpy.ndarray
        Array with shape (n_states_in_period, n_choices * 2 + 1) containing exogenous
        variables.
    max_value_functions : numpy.ndarray
        Array with shape (n_states_in_period,) containing the maximum over all value
        functions.
    not_interpolated : numpy.ndarray
        Array with shape (n_states_in_period,) containing indicator for states which
        are not interpolated and used to estimate the coefficients for the
        interpolation.

    """
    # Define ordinary least squares model and fit to the data.
    beta = ols(endogenous, exogenous[not_interpolated])

    # Use the model to predict EMAX for all states. As in Keane & Wolpin (1994),
    # negative predictions are truncated to zero.
    endogenous_predicted = exogenous.dot(beta)
    endogenous_predicted = clip(endogenous_predicted, 0)

    # Construct predicted EMAX for all states and the
    predictions = endogenous_predicted + max_value_functions
    predictions[not_interpolated] = endogenous + max_value_functions[not_interpolated]

    if not np.all(np.isfinite(beta)):
        warnings.warn("OLS coefficients in the interpolation are not finite.")

    return predictions


@nb.guvectorize(
    ["f8[:], f8[:], f8[:], f8[:, :], f8, b1[:], f8[:]"],
    "(n_choices), (n_choices), (n_choices), (n_draws, n_choices), (), (n_choices) "
    "-> ()",
    nopython=True,
    target="parallel",
)
def calculate_emax_value_functions(
    wages,
    nonpec,
    continuation_values,
    draws,
    delta,
    is_inadmissible,
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
    available in some choices. Missing choices are filled with ones. In the case of a
    choice with wage and without wage, flow utilities are

    .. math::

        \text{Flow Utility} = \text{Wage} * \epsilon + \text{Non-pecuniary}
        \text{Flow Utility} = 1 * \epsilon + \text{Non-pecuniary}


    Parameters
    ----------
    wages : numpy.ndarray
        Array with shape (n_choices,) containing wages.
    nonpec : numpy.ndarray
        Array with shape (n_choices,) containing non-pecuniary rewards.
    continuation_values : numpy.ndarray
        Array with shape (n_choices,) containing expected maximum utility for each
        choice in the subsequent period.
    draws : numpy.ndarray
        Array with shape (n_draws, n_choices).
    delta : float
        The discount factor.
    is_inadmissible: numpy.ndarray
        Array with shape (n_choices,) containing indicator for whether the following
        state is inadmissible.

    Returns
    -------
    emax_value_functions : float
        Expected maximum utility of an agent.

    .. _Monte Carlo integration:
        https://en.wikipedia.org/wiki/Monte_Carlo_integration

    """
    n_draws, n_choices = draws.shape

    emax_value_functions[0] = 0.0

    for i in range(n_draws):

        max_value_functions = 0.0

        for j in range(n_choices):
            value_function, _ = aggregate_keane_wolpin_utility(
                wages[j],
                nonpec[j],
                continuation_values[j],
                draws[i, j],
                delta,
                is_inadmissible[j],
            )

            if value_function > max_value_functions:
                max_value_functions = value_function

        emax_value_functions[0] += max_value_functions

    emax_value_functions[0] /= n_draws


@nb.njit
def ols(y, x):
    """Calculate OLS coefficients using a pseudo-inverse.

    Parameters
    ----------
    x : numpy.ndarray
        n x n matrix of independent variables.
    y : numpy.ndarray
        n x 1 matrix with dependent variable.

    Returns
    -------
    beta : numpy.ndarray
        n x 1 array of estimated parameter vector

    """
    beta = np.dot(np.linalg.pinv(x.T.dot(x)), x.T.dot(y))
    return beta


def mse(x1, x2, axis=0):
    """Calculate mean squared error.

    If ``x1`` and ``x2`` have different shapes, then they need to broadcast. This uses
    :func:`numpy.asanyarray` to convert the input. Whether this is the desired result or
    not depends on the array subclass, for example NumPy matrices will silently
    produce an incorrect result.

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two arrays.
    axis : int
       Axis along which the summary statistic is calculated

    Returns
    -------
    mse : numpy.ndarray or float
       Mean squared error along given axis.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1 - x2) ** 2, axis=axis)


def rmse(x1, x2, axis=0):
    """Calculate root mean squared error.

    If ``x1`` and ``x2`` have different shapes, then they need to broadcast. This uses
    :func:`numpy.asanyarray` to convert the input. Whether this is the desired result or
    not depends on the array subclass, for example NumPy matrices will silently
    produce an incorrect result.

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two arrays.
    axis : int
       Axis along which the summary statistic is calculated.

    Returns
    -------
    rmse : numpy.ndarray or float
       Root mean squared error along given axis.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))
