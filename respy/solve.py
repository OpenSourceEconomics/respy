"""Everything related to the solution of a structural model."""
import numba as nb

from respy.interpolate import interpolate
from respy.parallelization import parallelize_across_dense_dimensions
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import aggregate_keane_wolpin_utility
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
    state_space.create_choice_rewards(optim_paras, options)

    if optim_paras["delta"] == 0:
        solve_for_myopic_individuals(state_space)
    else:
        solve_with_backward_induction(state_space, optim_paras, options)

    return state_space


@parallelize_across_dense_dimensions()
def solve_for_myopic_individuals(state_space):
    """Solve the dynamic programming problem for myopic individuals."""
    state_space.get_attribute("expected_value_functions")[:] = 0


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
    n_wages = len(optim_paras["choices_w_wage"])
    n_periods = optim_paras["n_periods"]
    shocks_cholesky = optim_paras["shocks_cholesky"]

    draws_emax_risk = transform_base_draws_with_cholesky_factor(
        state_space.base_draws_sol, shocks_cholesky, n_wages
    )

    for period in reversed(range(n_periods)):
        slice_ = state_space.slices_by_periods[period]
        n_states_in_period = len(range(slice_.start, slice_.stop))

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In that case no interpolation is needed.
        any_interpolated = (
            options["interpolation_points"] <= n_states_in_period
            and options["interpolation_points"] != -1
        )

        if any_interpolated:
            interpolate(state_space, period, draws_emax_risk, optim_paras, options)

        else:
            _full_solution(state_space, period, draws_emax_risk, optim_paras)


@parallelize_across_dense_dimensions()
def _full_solution(state_space, period, draws_emax_risk, optim_paras):
    """Calculate the full solution of the model.

    In contrast to approximate solution, the Monte Carlo integration is done for each
    state and not only a subset.

    """
    slice_ = state_space.slices_by_periods[period]
    wages = state_space.get_attribute_from_period("wages", period)
    nonpec = state_space.get_attribute_from_period("nonpecs", period)
    is_inadmissible = state_space.get_attribute_from_period("is_inadmissible", period)
    continuation_values = state_space.get_continuation_values(period)

    exp_val_funcs = calculate_emax_value_functions(
        wages,
        nonpec,
        continuation_values,
        draws_emax_risk[period],
        optim_paras["delta"],
        is_inadmissible,
    )

    state_space.get_attribute("expected_value_functions")[slice_] = exp_val_funcs


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

    emax_value_functions[0] = 0

    for i in range(n_draws):

        max_value_functions = 0

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
