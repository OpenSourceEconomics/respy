from respy.python.solve.solve_auxiliary import (
    pyth_calculate_rewards_systematic
)
from respy.python.record.record_solution import record_solution_progress
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.shared.shared_auxiliary import create_covariates


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
    # Creating the state space of the model and collect the results in the
    # package class.
    record_solution_progress(1, file_sim)

    states = pyth_create_state_space(num_periods, num_types, edu_spec)

    states = create_covariates(states)

    record_solution_progress(-1, file_sim)

    # Calculate systematic rewards which are later used in the backward
    # induction procedure. These are calculated without any reference
    # to the alternative shock distributions.
    record_solution_progress(2, file_sim)

    # Calculate all systematic rewards
    states = pyth_calculate_rewards_systematic(
        states,
        optim_paras,
    )

    record_solution_progress(-1, file_sim)

    # Backward iteration procedure. There is a PYTHON and FORTRAN
    # implementation available. If agents are myopic, the backward induction
    # procedure is not called upon.
    record_solution_progress(3, file_sim)

    breakpoint()

    periods_emax = pyth_backward_induction(
        num_periods,
        is_myopic,
        periods_draws_emax,
        num_draws_emax,
        states,
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

    # Collect return arguments in tuple
    args = (
        periods_rewards_systematic,
        states_number_period,
        mapping_state_idx,
        periods_emax,
        states_all,
    )

    # Finishing
    return args
