#!/usr/bin/env python
""" This module allows to solve a dynamic programming model using the terminal.
"""

# standard library
import argparse
import sys
import os

# ROBUPY
sys.path.insert(0, os.environ['ROBUPY'])
from robupy.solve import solve as robupy_solve
from robupy.read import read


def _distribute_inputs(parser):
    """ Process input arguments.
    """
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    model = args.model

    # Assertions
    assert (os.path.exists(model))

    # Finishing.
    return model


def solve(model):
    """ Solve and simulate the dynamic programming model.
    """

    # Process initialization file
    robupy_obj = read(model)

    # Solve model
    robupy_solve(robupy_obj)

''' Execution of module as script.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Solve dynamic discrete choice model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', action='store', dest='model',
                        default='model.robupy.ini',
                        help='model initialization file')

    model = _distribute_inputs(parser)

    solve(model)
