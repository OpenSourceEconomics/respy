"""Everything related to the solution of a structural model."""
import functools

import numpy as np

from respy.exogenous_processes import compute_transition_probabilities
from respy.interpolate import kw_94_interpolation
from respy.parallelization import parallelize_across_dense_dimensions
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import calculate_expected_value_functions
from respy.shared import dump_objects
from respy.shared import load_objects
from respy.shared import pandas_dot
from respy.shared import select_valid_choices
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

    Examples
    --------
    >>> import respy as rp
    >>> params, options = rp.get_example_model("robinson_crusoe_basic", with_data=False)
    >>> solve = rp.get_solve_func(params, options)
    >>> state_space = solve(params)

    """
    optim_paras, options = process_params_and_options(params, options)

    state_space = create_state_space_class(optim_paras, options)
    solve_function = functools.partial(solve, options=options, state_space=state_space)

    return solve_function


def solve(params, options, state_space):
    """Solve the model."""
    optim_paras, options = process_params_and_options(params, options)

    transit_keys = None
    if hasattr(state_space, "dense_key_to_transit_keys"):
        transit_keys = state_space.dense_key_to_transit_keys

    dense_key_to_exogenous = None
    if hasattr(state_space, "dense_key_to_exogenous"):
        dense_key_to_exogenous = state_space.dense_key_to_exogenous

    wages, nonpecs = _create_param_specific_objects(
        state_space.dense_key_to_complex,
        state_space.dense_key_to_choice_set,
        optim_paras,
        options,
        transit_keys=transit_keys,
        bypass={"dense_key_to_exogenous": dense_key_to_exogenous},
    )

    state_space.wages = wages
    state_space.nonpecs = nonpecs

    state_space = _solve_with_backward_induction(state_space, optim_paras, options)

    return state_space


@parallelize_across_dense_dimensions
def _create_param_specific_objects(
    complex_,
    choice_set,
    optim_paras,
    options,
    dense_key_to_exogenous,
    transit_keys=None,
):
    """Create param specific objects.

    This function creates objects that are not fixed for a given model.
    Depending on their size they are either kept in working memory such
    as wages or dumped on disk such as transition probabilities!
    In the medium run we could also allow for fixed params here by saving values
    on disk directly!
    For objects that we store on disk we will just return the prefix of the location.
    """
    states = load_objects("states", complex_, options)
    wages, nonpecs = _create_choice_rewards(states, choice_set, optim_paras)

    if len(optim_paras["exogenous_processes"]) > 0:
        transition_probabilities = compute_transition_probabilities(
            states, transit_keys, optim_paras, dense_key_to_exogenous
        )
        dump_objects(transition_probabilities, "transition", complex_, options)

    return wages, nonpecs


def _create_choice_rewards(states, choice_set, optim_paras):
    """Create wage and non-pecuniary reward for each state and choice."""
    n_choices = sum(choice_set)
    choices = select_valid_choices(optim_paras["choices"], choice_set)

    n_states = states.shape[0]

    wages = np.ones((n_states, n_choices))
    nonpecs = np.zeros((n_states, n_choices))

    for i, choice in enumerate(choices):
        if f"wage_{choice}" in optim_paras:
            log_wage = pandas_dot(states, optim_paras[f"wage_{choice}"])
            wages[:, i] = np.exp(log_wage)

        if f"nonpec_{choice}" in optim_paras:
            nonpecs[:, i] = pandas_dot(states, optim_paras[f"nonpec_{choice}"])

    return wages, nonpecs


def _solve_with_backward_induction(state_space, optim_paras, options):
    """Calculate utilities with backward induction.

    The expected value functions in one period are only computed by interpolation if:

    1. Interpolation is requested.
    2. If there are more states in the period than interpolation points.
    3. If there are at least two interpolation points per `dense_index`.

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
    n_periods = options["n_periods"]

    # Can we move that up to the other function? Then we have everything at one point
    # that is params specific?
    draws_emax_risk = transform_base_draws_with_cholesky_factor(
        state_space.base_draws_sol,
        state_space.dense_key_to_choice_set,
        optim_paras["shocks_cholesky"],
        optim_paras,
    )

    for period in reversed(range(n_periods)):
        dense_keys_in_period = state_space.get_dense_keys_from_period(period)

        period_draws_emax_risk = {
            dense_index: draws_emax_risk[dense_index]
            for dense_index in dense_keys_in_period
        }

        n_states_in_period = sum(
            len(state_space.dense_key_to_core_indices[dense_index])
            for dense_index in dense_keys_in_period
        )

        # See docstring for note on interpolation.
        any_interpolated = options[
            "interpolation_points"
        ] < n_states_in_period and options["interpolation_points"] >= 2 * len(
            dense_keys_in_period
        )

        # Handle myopic individuals. Check interpolation!
        if optim_paras["delta"] == 0:
            period_expected_value_functions = {k: 0 for k in dense_keys_in_period}

        elif any_interpolated:
            period_expected_value_functions = kw_94_interpolation(
                state_space, period_draws_emax_risk, period, optim_paras, options,
            )

        else:

            wages = state_space.get_attribute_from_period("wages", period)
            nonpecs = state_space.get_attribute_from_period("nonpecs", period)
            continuation_values = state_space.get_continuation_values(period)
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
    state and not only a subset of states.

    """
    period_expected_value_functions = calculate_expected_value_functions(
        wages,
        nonpecs,
        continuation_values,
        period_draws_emax_risk,
        optim_paras["delta"],
    )

    return period_expected_value_functions
