#!/usr/bin/env python

# standard library
import pickle as pkl
import numpy as np

import argparse
import os

# project library
from respy.python.estimate.estimate_auxiliary import dist_optim_paras

from respy import simulate
from respy import RespyCls


def dist_input_arguments(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    init_file = args.init_file
    file_sim = args.file_sim
    update = args.update
    solved = args.solved

    # Check attributes
    assert (update in [False, True])
    assert (os.path.exists(init_file))

    if update:
        assert (os.path.exists('paras_steps.respy.log'))

    if solved is not None:
        assert (os.path.exists(solved))
        assert (update is False)

    # Finishing
    return update, init_file, file_sim, solved


def scripts_simulate(update, init_file, file_sim, solved):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification.
    if solved is not None:
        respy_obj = pkl.load(open(solved, 'rb'))
    else:
        respy_obj = RespyCls(init_file)

    # Update parametrization of the model if resuming from a previous
    # estimation run.
    if update:
        x0 = np.genfromtxt('paras_steps.respy.log')
        respy_obj.update_model_paras(x0)

    # Update file for output.
    if file_sim is not None:
        respy_obj.unlock()
        respy_obj.set_attr('file_sim', file_sim)
        respy_obj.lock()

    # Optimize the criterion function.
    simulate(respy_obj)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Start of simulation with the RESPY package.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--update', action='store_true', dest='update',
        default=False, help='update model parametrization')

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.respy.ini', help='initialization file')

    parser.add_argument('--file_sim', action='store', dest='file_sim',
        default=None, help='output file')

    parser.add_argument('--solved', action='store', dest='solved',
        default=None, help='use solved class instance')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run simulation
    scripts_simulate(*args)
