#!/usr/bin/env python
""" This script serves as a command line tool to ease the estimation of the
model.
"""

# standard library
import numpy as np

import argparse
import os

# project library
from robupy.estimate.estimate_auxiliary import opt_get_model_parameters
from robupy.estimate.estimate import estimate
from robupy.process.process import process
from robupy.read.read import read

""" Auxiliary function
"""


def distribute_input_arguments(parser):
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
        assert (os.path.exists('paras_steps.robupy.log'))

    # Finishing
    return resume, single, init_file


""" Main function
"""


def estimate_wrapper(resume, single, init_file):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification.
    robupy_obj = read(init_file)

    # Update parametrization of the model if resuming from a previous
    # estimation run.
    if resume:
        x0 = np.genfromtxt('paras_steps.robupy.log')
        args = opt_get_model_parameters(x0, True)[:-1]
        robupy_obj.update_model_paras(*args)

    # Set maximum iteration count when only an evaluation of the criterion
    # function is requested.
    if single:
        robupy_obj.unlock()
        robupy_obj.set_attr('maxiter', 0)
        robupy_obj.lock()

    # Process dataset
    data_frame = process(robupy_obj)

    # Optimize the criterion function.
    estimate(robupy_obj, data_frame)

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description =
        'Start of estimation run with the ROBUPY package.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--resume', action ='store_true',  dest='resume',
        default=False, help='resume estimation run')

    parser.add_argument('--single', action='store_true', dest='single',
        default=False, help='single evaluation')

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.robupy.ini', help='initialization file')

    resume, single, init_file = distribute_input_arguments(parser)

    estimate_wrapper(resume, single, init_file)