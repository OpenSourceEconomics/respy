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
    init_file = args.init_file
    resume = args.resume
    single = args.single

    # Check attributes
    assert (single in [True, False])
    assert (resume in [False, True])
    assert (os.path.exists(init_file))

    if resume:
        assert (os.path.exists('est.respy.info'))

    # Finishing
    return resume, single, init_file


def scripts_estimate(resume, single, init_file):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification.
    respy_obj = RespyCls(init_file)

    # Update parametrization of the model if resuming from a previous
    # estimation run.
    if resume:
        respy_obj.update_model_paras(get_est_info()['paras_step'])

    # Set maximum iteration count when only an evaluation of the criterion
    # function is requested.
    if single:
        respy_obj.unlock()
        respy_obj.set_attr('maxfun', 0)
        respy_obj.lock()

    # Optimize the criterion function.
    estimate(respy_obj)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Start of estimation run with the RESPY package.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--resume', action='store_true', dest='resume',
        default=False, help='resume estimation run')

    parser.add_argument('--single', action='store_true', dest='single',
        default=False, help='single evaluation')

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.respy.ini', help='initialization file')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run estimation
    scripts_estimate(*args)
