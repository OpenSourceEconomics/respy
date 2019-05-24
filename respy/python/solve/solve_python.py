from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.solve.solve_auxiliary import StateSpace


def pyth_solve(
    interpolation,
    num_points_interp,
    num_periods,
    periods_draws_emax,
    edu_spec,
    optim_paras,
    num_types,
):
    """Solve the model.

    This function is a wrapper around state space creation and determining the optimal
    decision in each state by backward induction.

    Parameters
    ----------
    interpolation : bool
        Indicator for whether the expected maximum utility should be interpolated.
    num_points_interp : int
        Number of points used for the interpolation.
    num_periods : int
        Number of periods.
    periods_draws_emax : np.ndarray
        Array with shape (num_periods, num_draws, num_choices) containing draws for the
        Monte Carlo simulation of expected maximum utility.
    edu_spec : dict
        Information on education.
    optim_paras : dict
        Parameters affected by optimization.
    num_types : int
        Number of types.

    """
    state_space = StateSpace(
        num_periods, num_types, edu_spec["start"], edu_spec["max"], optim_paras
    )

    state_space = pyth_backward_induction(
        periods_draws_emax, state_space, interpolation, num_points_interp, optim_paras
    )

    return state_space
