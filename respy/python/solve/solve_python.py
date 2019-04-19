from respy.python.record.record_solution import record_solution_progress
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.solve.solve_auxiliary import StateSpace


def pyth_solve(
    is_interpolated,
    num_points_interp,
    num_periods,
    is_debug,
    periods_draws_emax,
    edu_spec,
    optim_paras,
    file_sim,
    num_types,
):
    """Solve the model.

    This function is a wrapper around state space creation and determining the optimal
    decision in each state by backward induction.

    Parameters
    ----------
    is_interpolated : bool
        Indicator for whether the expected maximum utility should be interpolated.
    num_points_interp : int
        Number of points used for the interpolation.
    num_periods : int
        Number of periods.
    is_debug : bool
        Flag for debugging.
    periods_draws_emax : np.ndarray
        Array with shape (num_periods, num_draws, num_choices) containing draws for the
        Monte Carlo simulation of expected maximum utility.
    edu_spec : dict
    optim_paras : dict
    file_sim : ???
    num_types : int
        Number of types.

    """
    record_solution_progress(1, file_sim)

    # Create the state space
    state_space = StateSpace(
        num_periods, num_types, edu_spec["start"], edu_spec["max"], optim_paras
    )

    record_solution_progress(-1, file_sim)

    record_solution_progress(2, file_sim)

    record_solution_progress(-1, file_sim)

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available. If agents are myopic, the backward induction
    # procedure is not called upon.
    record_solution_progress(3, file_sim)

    state_space = pyth_backward_induction(
        periods_draws_emax,
        state_space,
        is_debug,
        is_interpolated,
        num_points_interp,
        optim_paras,
        file_sim,
        True,
    )

    if optim_paras["delta"]:
        record_solution_progress(-1, file_sim)

    return state_space
