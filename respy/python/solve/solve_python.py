from respy.python.record.record_solution import record_solution_progress
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.solve.solve_auxiliary import StateSpace
from respy.python.solve.solve_auxiliary import (
    pyth_calculate_rewards_systematic,
)


def pyth_solve(
    is_interpolated,
    num_points_interp,
    num_draws_emax,
    num_periods,
    is_myopic,
    is_debug,
    periods_draws_emax,
    edu_spec,
    optim_paras,
    file_sim,
    optimizer_options,
    num_types,
):
    """ Solving the model using pure PYTHON code.
    """
    record_solution_progress(1, file_sim)

    # Create the state space
    state_space = StateSpace(num_periods, num_types, edu_spec["start"], edu_spec["max"])

    record_solution_progress(-1, file_sim)

    record_solution_progress(2, file_sim)

    state_space.states = pyth_calculate_rewards_systematic(
        state_space.states, optim_paras
    )

    record_solution_progress(-1, file_sim)

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available. If agents are myopic, the backward induction
    # procedure is not called upon.
    record_solution_progress(3, file_sim)

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
        file_sim,
        True,
    )

    if not is_myopic:
        record_solution_progress(-1, file_sim)

    return state_space
