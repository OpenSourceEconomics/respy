#!/usr/bin/env python
import argparse
import os

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

    # Check attributes
    assert (os.path.exists(init_file))

    # Finishing
    return init_file, file_sim


def scripts_simulate(init_file, file_sim):
    """ Wrapper for the estimation.
    """
    respy_obj = RespyCls(init_file)

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

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.respy.ini', help='initialization file')

    parser.add_argument('--file_sim', action='store', dest='file_sim',
        default=None, help='output file')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run simulation
    scripts_simulate(*args)
