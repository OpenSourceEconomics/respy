#!/usr/bin/env python
import argparse
import os

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
    # Read in baseline model specification.
    respy_obj = RespyCls(init_file)

    if request == 'estimate':
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
