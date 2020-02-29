"""Everything related to the solution of a structural model."""
import functools

import numpy as np

from respy.config import COVARIATES_DOT_PRODUCT_DTYPE
from respy.interpolate import interpolate
from respy.parallelization import parallelize_across_dense_dimensions
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import calculate_expected_value_functions
from respy.shared import transform_base_draws_with_cholesky_factor
from respy.state_space import create_state_space_class


def get_solve_func(params, options):
    """Get the solve function.

    This function takes a model specification and returns the state space of the model
    along with components of the solution such as covariates, non-pecuniary rewards,
    wages, continuation values and expected value functions as attributes of the class.

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame containing parameter series.
    options : dict
        Dictionary containing model attributes which are not optimized.

    Returns
    -------
    solve : :func:`~respy.solve.solve`
        Function with partialed arguments.

    """
    optim_paras, options = process_params_and_options(params, options)

    state_space = create_state_space_class(optim_paras, options)
    solve_function = functools.partial(solve, options=options, state_space=state_space)

    return solve_function


def solve(params, options, state_space):
    """Solve the model."""
    optim_paras, options = process_params_and_options(params, options)

    states = state_space.states
    wages = state_space.get_attribute("wages")
    nonpecs = state_space.get_attribute("nonpecs")

    wages, nonpecs = _create_choice_rewards(states, wages, nonpecs, optim_paras)
    state_space.set_attribute("wages", wages)
    state_space.set_attribute("nonpecs", nonpecs)

    if optim_paras["delta"] == 0:
        expected_value_functions = _solve_for_myopic_individuals(
            state_space.get_attribute("expected_value_functions")
        )
        state_space.set_attribute("expected_value_functions", expected_value_functions)
    else:
        state_space = _solve_with_backward_induction(state_space, optim_paras, options)

    return state_space


@parallelize_across_dense_dimensions
def _create_choice_rewards(states, wages, nonpecs, optim_paras):
    """Create wage and non-pecuniary reward for each state and choice.

    Note that missing wages filled with ones and missing non-pecuniary rewards with
    zeros. This is done in :meth:`_initialize_attributes`.

    """
    for i, choice in enumerate(optim_paras["choices"]):
        if f"wage_{choice}" in optim_paras:
            wage_columns = optim_paras[f"wage_{choice}"].index
            log_wage = np.dot(
                states[wage_columns].to_numpy(dtype=COVARIATES_DOT_PRODUCT_DTYPE),
                optim_paras[f"wage_{choice}"].to_numpy(),
            )
            wages[:, i] = np.exp(log_wage)

        if f"nonpec_{choice}" in optim_paras:
            nonpec_columns = optim_paras[f"nonpec_{choice}"].index
            nonpecs[:, i] = np.dot(
                states[nonpec_columns].to_numpy(dtype=COVARIATES_DOT_PRODUCT_DTYPE),
                optim_paras[f"nonpec_{choice}"].to_numpy(),
            )

    return wages, nonpecs


@parallelize_across_dense_dimensions
def _solve_for_myopic_individuals(expected_value_functions):
    """Solve the dynamic programming problem for myopic individuals."""
    expected_value_functions[:] = 0
    return expected_value_functions


def _solve_with_backward_induction(state_space, optim_paras, options):
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
    n_wages = len(optim_paras["choices_w_wage"])
    n_periods = optim_paras["n_periods"]
    shocks_cholesky = optim_paras["shocks_cholesky"]

    draws_emax_risk = transform_base_draws_with_cholesky_factor(
        state_space.base_draws_sol, shocks_cholesky, n_wages
    )

    for period in reversed(range(n_periods)):
        wages = state_space.get_attribute_from_period("wages", period)
        nonpecs = state_space.get_attribute_from_period("nonpecs", period)
        is_inadmissible = state_space.get_attribute_from_period(
            "is_inadmissible", period
        )
        continuation_values = state_space.get_continuation_values(period)
        period_draws_emax_risk = draws_emax_risk[period]

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In this case, no interpolation is needed.
        n_dense_combinations = len(getattr(state_space, "sub_state_spaces", [1]))
        n_core_states = state_space.core.query("period == @period").shape[0]
        n_states_in_period = n_core_states * n_dense_combinations
        any_interpolated = (
            options["interpolation_points"] <= n_states_in_period
            and options["interpolation_points"] != -1
        )

        if any_interpolated:
            interp_points = int(options["interpolation_points"] / n_dense_combinations)
            period_expected_value_functions = interpolate(
                wages,
                nonpecs,
                continuation_values,
                is_inadmissible,
                period_draws_emax_risk,
                interp_points,
                optim_paras,
                options,
            )

        else:
            period_expected_value_functions = _full_solution(
                wages,
                nonpecs,
                continuation_values,
                is_inadmissible,
                period_draws_emax_risk,
                optim_paras,
            )

        state_space.set_attribute_from_period(
            "expected_value_functions", period_expected_value_functions, period
        )

    return state_space


@parallelize_across_dense_dimensions
def _full_solution(
    wages,
    nonpecs,
    continuation_values,
    is_inadmissible,
    period_draws_emax_risk,
    optim_paras,
):
    """Calculate the full solution of the model.

    In contrast to approximate solution, the Monte Carlo integration is done for each
    state and not only a subset.

    """
    period_expected_value_functions = calculate_expected_value_functions(
        wages,
        nonpecs,
        continuation_values,
        period_draws_emax_risk,
        optim_paras["delta"],
        is_inadmissible,
    )

    return period_expected_value_functions
