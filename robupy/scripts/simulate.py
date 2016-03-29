#!/usr/bin/env python
""" This script serves as a command line tool to ease the simulation of the
model.
"""

# standard library
import numpy as np

import argparse
import os

# project library
from robupy.estimate.estimate_auxiliary import opt_get_model_parameters
from robupy.simulate.simulate import simulate
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
    update = args.update

    # Check attributes
    assert (update in [False, True])
    assert (os.path.exists(init_file))

    if update:
        assert (os.path.exists('steps_paras.robupy.log'))

    # Finishing
    return update, init_file


""" Main function
"""


def simulate_wrapper(update, init_file):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification.
    robupy_obj = read(init_file)

    # Update parametrization of the model if resuming from a previous
    # estimation run.
    if update:
        x0 = np.genfromtxt('steps_paras.robupy.log')
        args = opt_get_model_parameters(x0, True)[:-1]
        robupy_obj.update_model_paras(*args)

   # Optimize the criterion function.
    simulate(robupy_obj)


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description =
        'Start of simulation with the ROBUPY package.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--update', action ='store_true',  dest='update',
        default=False, help='update model parametrization')

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.robupy.ini', help='initialization file')

    update, init_file = distribute_input_arguments(parser)

    simulate_wrapper(update, init_file)