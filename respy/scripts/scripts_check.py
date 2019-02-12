#!/usr/bin/env python
import numpy as np

import argparse
import os

from respy.python.solve.solve_auxiliary import StateSpace
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.pre_processing.data_processing import process_dataset
from respy.custom_exceptions import UserError
from respy import RespyCls

# module-wide variables
ERR_MSG = " Observations not meeting model requirements."


def dist_input_arguments(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    init_file = args.init_file
    request = args.request

    # Check attributes
    assert os.path.exists(init_file)
    assert request in ["estimate", "simulate"]

    # Finishing
    return request, init_file


def scripts_check(request, init_file):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification
    respy_obj = RespyCls(init_file)

    # Distribute model parameters
    num_periods, edu_spec, num_types = dist_class_attributes(
        respy_obj, "num_periods", "edu_spec", "num_types"
    )

    # We need to run additional checks if an estimation is requested.
    if request == "estimate":
        # Create the grid of the admissible states.
        state_space = StateSpace()
        state_space.create_state_space(num_periods, num_types, edu_spec)

        # We also check the structure of the dataset.
        data_array = process_dataset(respy_obj).values
        num_rows = data_array.shape[0]

        for j in range(num_rows):
            period = int(data_array[j, 1])
            # Extract observable components of state space as well as agent decision.
            exp_a, exp_b, edu, choice_lagged = data_array[j, 4:].astype(int)

            # First of all, we need to ensure that all observed years of schooling are larger
            # than the initial condition of the model.
            try:
                np.testing.assert_equal(edu >= 0, True)
            except AssertionError:
                raise UserError(ERR_MSG)

            # Get state indicator to obtain the systematic component of the agents rewards. This
            # might fail either because the state is simply infeasible at any period or just not
            # defined for the particular period requested.
            try:
                k = state_space[period, exp_a, exp_b, edu, choice_lagged - 1]
                np.testing.assert_equal(k >= 0, True)
            except (IndexError, AssertionError):
                raise UserError(ERR_MSG)

        # We also take a special look at the optimizer options.
        respy_obj.check_estimation()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check request for the RESPY package.")

    parser.add_argument(
        "--request",
        action="store",
        dest="request",
        help="task to perform",
        required=True,
    )

    parser.add_argument(
        "--init",
        action="store",
        dest="init_file",
        default="model.respy.ini",
        help="initialization file",
    )

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run estimation
    scripts_check(*args)
