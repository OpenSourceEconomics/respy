#!/usr/bin/env python

import numpy as np
import argparse
import os

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.estimate.estimate_auxiliary import get_optim_paras
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import get_est_info
from respy import estimate
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
        file_est = respy_obj.get_attr('file_est')
        assert os.path.exists(file_est)

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
