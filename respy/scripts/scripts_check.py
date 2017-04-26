#!/usr/bin/env python
import numpy as np

import argparse
import os

from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.process.process_python import process
from respy.estimate import check_estimation
from respy import RespyCls

# module-wide variables
ERR_MSG = ' Observation not meeting model requirements.'


def dist_input_arguments(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    request = args.request
    init = args.init

    # Check attributes
    assert (os.path.exists(init))
    assert (request in ['estimate', 'simulate'])

    # Finishing
    return request, init


def scripts_check(request, init_file):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification
    respy_obj = RespyCls(init_file)

    # Distribute model parameters
    num_periods = respy_obj.get_attr('num_periods')
    edu_start = respy_obj.get_attr('edu_start')
    edu_max = respy_obj.get_attr('edu_max')
    min_idx = respy_obj.get_attr('min_idx')

    # We need to run additional checks if an estimation is requested.
    if request == 'estimate':
        # Create the grid of the admissible states.
        args = (num_periods, edu_start, edu_max, min_idx)
        mapping_state_idx = pyth_create_state_space(*args)[2]

        # We also check the structure of the dataset.
        data_array = process(respy_obj).as_matrix()
        num_rows = data_array.shape[0]

        for j in range(num_rows):
            period = int(data_array[j, 1])
            # Extract observable components of state space as well as agent
            # decision.
            exp_a, exp_b, edu, edu_lagged = data_array[j, 4:].astype(int)
            edu = edu - edu_start

            # First of all, we need to ensure that all observed years of
            # schooling are larger than the initial condition of the model.
            try:
                np.testing.assert_equal(edu >= 0, True)
            except AssertionError:
                raise AssertionError(ERR_MSG)

            # Get state indicator to obtain the systematic component of the
            # agents rewards. This might fail either because the state is
            # simply infeasible at any period or just not defined for the
            # particular period requested.
            try:
                k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]
                np.testing.assert_equal(k >= 0, True)
            except (IndexError, AssertionError):
                raise AssertionError(ERR_MSG)

        # We also take a special look at the optimizer options.
        check_estimation(respy_obj)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Check request for the RESPY package.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--request', action='store', dest='request',
        help='task to perform', required=True)

    parser.add_argument('--init', action='store', dest='init',
        default='model.respy.ini', help='initialization file')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run estimation
    scripts_check(*args)
