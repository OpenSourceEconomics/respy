"""This module contains the code for approximate solutions to the DCDP."""
import warnings

import numba as nb
import numpy as np

from respy.config import MAX_LOG_FLOAT
from respy.parallelization import parallelize_across_dense_dimensions
from respy.shared import calculate_expected_value_functions
from respy.shared import calculate_value_functions_and_flow_utilities


@parallelize_across_dense_dimensions
def interpolate(
    wages,
    nonpecs,
    continuation_values,
    is_inadmissible,
    period_draws_emax_risk,
    interpolation_points,
    optim_paras,
    options,
):
    """Interface to switch between different interpolation routines."""
    period_expected_value_functions = _kw_94_interpolation(
        wages,
        nonpecs,
        continuation_values,
        is_inadmissible,
        period_draws_emax_risk,
        interpolation_points,
        optim_paras,
        options,
    )

    return period_expected_value_functions


def _kw_94_interpolation(
    wages,
    nonpecs,
    continuation_values,
    is_inadmissible,
    period_draws_emax_risk,
    interpolation_points,
    optim_paras,
    options,
):
    """Calculate the approximate solution proposed by [1]_.

    The authors propose an interpolation method to alleviate the computation burden of
    the full solution. The full solution calculates the expected value function with
    Monte-Carlo simulation for each state in the state space for a pre-defined number of
    points. Both, the number of states and points, have a huge impact on runtime.

    [1]_ propose an interpolation method to alleviate the computation burden. The
    general idea is to calculate the expected value function with Monte-Carlo simulation
    only for a much smaller number of states and predict the remaining expected value
    functions with a linear model.

    The function consists of the following steps.

    1. Create an indicator for whether the expected value function of the state is
       calculated with Monte-Carlo simulation or interpolation.

    2. Compute

    References
    ----------
    .. [1] Keane, M. P. and  Wolpin, K. I. (1994). `The Solution and Estimation of
           Discrete Choice Dynamic Programming Models by Simulation and Interpolation:
           Monte Carlo Evidence <https://doi.org/10.2307/2109768>`_. *The Review of
           Economics and Statistics*, 76(4): 648-672.



    """
    n_choices_w_wage = len(optim_paras["choices_w_wage"])
    n_core_states_in_period = wages.shape[0]

    # Get indicator for interpolation and simulation of states. The seed value is the
    # base seed plus the number of the period. Thus, not interpolated states are held
    # constant for each periods and not across periods.
    not_interpolated = _get_not_interpolated_indicator(
        interpolation_points,
        n_core_states_in_period,
        next(options["solution_seed_iteration"]),
    )

    # These shifts are used to determine the expected values of the working
    # alternatives. These are log normal distributed and thus the draws cannot simply
    # set to zero, but :math:`E(X) = \exp\{\mu + \frac{\sigma^2}{2}\}`.
    shifts = np.zeros(len(optim_paras["choices"]))
    shocks_cov = optim_paras["shocks_cholesky"].dot(optim_paras["shocks_cholesky"].T)
    shifts[:n_choices_w_wage] = np.exp(
        np.clip(np.diag(shocks_cov)[:n_choices_w_wage], 0, MAX_LOG_FLOAT) / 2
    )

    # Constructing the exogenous variable for all states, including the ones where
    # simulation will take place. All information will be used in either the
    # construction of the prediction model or the prediction step.
    exogenous, max_emax = _calculate_exogenous_variables(
        wages,
        nonpecs,
        continuation_values,
        shifts,
        optim_paras["delta"],
        is_inadmissible,
    )

    # Constructing the dependent variables for all states at the random subset of points
    # where the EMAX is actually calculated.
    endogenous = _calculate_endogenous_variables(
        wages,
        nonpecs,
        continuation_values,
        max_emax,
        not_interpolated,
        period_draws_emax_risk,
        optim_paras["delta"],
        is_inadmissible,
    )

    # Create prediction model based on the random subset of points where the EMAX is
    # actually simulated and thus dependent and independent variables are available. For
    # the interpolation points, the actual values are used.
    period_expected_value_functions = get_predictions(
        endogenous, exogenous, max_emax, not_interpolated
    )

    return period_expected_value_functions


def _get_not_interpolated_indicator(interpolation_points, n_states, seed):
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
    not_interpolated = np.zeros(n_states, dtype="bool")
    not_interpolated[indices] = True

    return not_interpolated


def _calculate_exogenous_variables(wages, nonpec, emaxs, draws, delta, is_inadmissible):
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


def _calculate_endogenous_variables(
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
    emax_value_functions = calculate_expected_value_functions(
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
    endogenous_predicted = np.clip(endogenous_predicted, 0, None)

    # Construct predicted EMAX for all states and the
    predictions = endogenous_predicted + max_value_functions
    predictions[not_interpolated] = endogenous + max_value_functions[not_interpolated]

    if not np.all(np.isfinite(beta)):
        warnings.warn("OLS coefficients in the interpolation are not finite.")

    return predictions


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
