from respy.python.solve.solve_auxiliary import (
    pyth_calculate_rewards_systematic,
)
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.shared.shared_auxiliary import distribute_parameters
from respy.python.shared.shared_auxiliary import get_log_likl


def pyth_criterion(
    x,
    is_interpolated,
    num_draws_emax,
    num_periods,
    num_points_interp,
    is_myopic,
    is_debug,
    data,
    num_draws_prob,
    tau,
    periods_draws_emax,
    periods_draws_prob,
    state_space,
    num_agents_est,
    num_obs_agent,
    num_types,
    edu_spec,
):
    """Criterion function for the likelihood maximization."""
    optim_paras = distribute_parameters(x, is_debug)

    # Calculate all systematic rewards
    state_space.states = pyth_calculate_rewards_systematic(
        state_space.states, optim_paras
    )

    state_space = pyth_backward_induction(
        num_periods,
        is_myopic,
        periods_draws_emax,
        num_draws_emax,
        state_space,
        is_debug,
        is_interpolated,
        num_points_interp,
        edu_spec,
        optim_paras,
        "",
        False,
    )

    contribs = pyth_contributions(
        state_space,
        data,
        periods_draws_prob,
        tau,
        num_periods,
        num_draws_prob,
        num_agents_est,
        num_obs_agent,
        num_types,
        edu_spec,
        optim_paras,
    )

    crit_val = get_log_likl(contribs)

    return crit_val
