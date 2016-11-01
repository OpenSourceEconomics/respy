#!/usr/bin/env python
import argparse
import os

from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.process.process_python import process
from respy.estimate import check_estimation
from respy import RespyCls


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
        num_obs = data_array.shape[0]

        for j in range(num_obs):
            period = int(data_array[j, 1])
            # Extract observable components of state space as well as agent
            # decision.
            exp_a, exp_b, edu, edu_lagged = data_array[j, 4:].astype(int)
            edu = edu - edu_start

            # Get state indicator to obtain the systematic component of the
            # agents rewards.
            try:
                mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]
            except IndexError:
                str_ = ' Observation not meeting model requirements.'
                raise AssertionError(str_)

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
