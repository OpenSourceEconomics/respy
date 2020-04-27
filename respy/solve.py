"""Everything related to the solution of a structural model."""
import functools

import numpy as np

from respy.interpolate import interpolate
from respy.parallelization import parallelize_across_dense_dimensions
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import calculate_expected_value_functions
from respy.shared import compute_covariates
from respy.shared import pandas_dot
from respy.shared import subset_to_period
from respy.state_space import create_state_space_class
from respy.state_space import transform_base_draws_with_cholesky_factor


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

    wages, nonpecs = _create_choice_rewards(
        state_space.core,
        state_space.index_to_indices,
        state_space.index_to_choice_set,
        state_space.index_to_dense_covariates,
        optim_paras,
        options,
    )

    state_space.set_attribute("wages", wages)
    state_space.set_attribute("nonpecs", nonpecs)

    state_space = _solve_with_backward_induction(state_space, optim_paras, options)

    return state_space


@parallelize_across_dense_dimensions
def _create_choice_rewards(
    core, core_indices, choice_set, dense_covariates, optim_paras, options
):
    """Create wage and non-pecuniary reward for each state and choice.

    TODO: The function receives the full core and the indices
     instead of the already indexed core
    which would require less memory to be copied and moved.
    At the same time, indexing and copying
    in the main process is also expensive.

    """
    n_choices = sum(choice_set)
    choices = [
        choice for i, choice in enumerate(optim_paras["choices"]) if choice_set[i]
    ]

    dense = core.loc[core_indices].copy().assign(**dense_covariates)
    states = compute_covariates(dense, options["covariates_all"])

    n_states = states.shape[0]

    wages = np.ones((n_states, n_choices))
    nonpecs = np.zeros((n_states, n_choices))

    for i, choice in enumerate(choices):
        if f"wage_{choice}" in optim_paras:
            pandas_dot(states, optim_paras[f"wage_{choice}"], out=wages[:, i])
            np.exp(wages[:, i], out=wages[:, i])

        if f"nonpec_{choice}" in optim_paras:
            pandas_dot(states, optim_paras[f"nonpec_{choice}"], out=nonpecs[:, i])

    return wages, nonpecs


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
    # Is this still practicable?
    n_periods = optim_paras["n_periods"]

    # Rewrite
    draws_emax_risk = transform_base_draws_with_cholesky_factor(
        state_space.base_draws_sol,
        state_space.index_to_choice_set,
        optim_paras["shocks_cholesky"],
        optim_paras,
    )

    for period in reversed(range(n_periods)):
        n_core_states = state_space.core.query("period == @period").shape[0]

        wages = state_space.get_attribute_from_period("wages", period)
        nonpecs = state_space.get_attribute_from_period("nonpecs", period)
        continuation_values = state_space.get_continuation_values(period)

        period_draws_emax_risk = subset_to_period(
            draws_emax_risk, state_space.index_to_complex, period
        )

        # The number of interpolation points is the same for all periods. Thus, for
        # some periods the number of interpolation points is larger than the actual
        # number of states. In this case, no interpolation is needed.
        n_dense_combinations = len(getattr(state_space, "sub_state_spaces", [1]))
        n_states_in_period = n_core_states * n_dense_combinations
        any_interpolated = (
            options["interpolation_points"] <= n_states_in_period
            and options["interpolation_points"] != -1
        )

        # Handle myopic individuals.
        if optim_paras["delta"] == 0:
            if hasattr(state_space, "sub_state_spaces"):
                period_expected_value_functions = {
                    dense_idx: {key: 0 for key in wages.keys()}
                    for dense_idx in state_space.sub_state_spaces
                }
            else:
                period_expected_value_functions = 0

        elif any_interpolated:
            period_expected_value_functions = interpolate(
                wages,
                nonpecs,
                continuation_values,
                state_space,
                period_draws_emax_risk,
                period,
                optim_paras,
                options,
            )

        else:
            period_expected_value_functions = _full_solution(
                wages, nonpecs, continuation_values, period_draws_emax_risk, optim_paras
            )

        state_space.set_attribute_from_keys(
            "expected_value_functions", period_expected_value_functions
        )

    return state_space


@parallelize_across_dense_dimensions
def _full_solution(
    wages, nonpecs, continuation_values, period_draws_emax_risk, optim_paras
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
    )

    return period_expected_value_functions
