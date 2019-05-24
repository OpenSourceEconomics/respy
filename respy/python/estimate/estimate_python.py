from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.shared.shared_auxiliary import distribute_parameters
from respy.python.shared.shared_auxiliary import get_log_likl
from respy.python.solve.solve_auxiliary import pyth_backward_induction


def pyth_criterion(
    x,
    interpolation,
    num_points_interp,
    is_debug,
    data,
    tau,
    periods_draws_emax,
    periods_draws_prob,
    state_space,
):
    """Criterion function for the likelihood maximization."""
    optim_paras = distribute_parameters(x, is_debug)

    # Calculate all systematic rewards
    state_space.update_systematic_rewards(optim_paras)

    state_space = pyth_backward_induction(
        periods_draws_emax, state_space, interpolation, num_points_interp, optim_paras
    )

    contribs = pyth_contributions(
        state_space, data, periods_draws_prob, tau, optim_paras
    )

    crit_val = get_log_likl(contribs)

    return crit_val
