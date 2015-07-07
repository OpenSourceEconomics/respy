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
from robupy.simulate import simulate as robupy_simulate
from robupy.read import read

def _distribute_inputs(parser):
    """ Process input arguments.
    """
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    simulate = args.simulate
    solution = args.solution
    model = args.model

    # Assertions.
    assert (simulate in [True, False])
    assert (os.path.exists(model))
    assert (isinstance(solution, str))

    # Finishing.
    return model, solution, simulate

def solve(model, solution, simulate):
    """ Solve the dynamic programming model.
    """

    # Process initialization file
    robupy_obj = read(model)

    # Solve model
    robupy_obj = robupy_solve(robupy_obj)

    robupy_obj.store(solution)

    # Simulate the model
    if simulate:
        robupy_simulate(robupy_obj)

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Solve dynamic discrete choice model.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', action='store', dest='model',
                        default='model.robupy.ini',
                        help='model initialization file')

    parser.add_argument('--simulate', action='store_true', dest='simulate',
                        default=False, help='simulate')

    parser.add_argument('--solution', action='store', dest='solution',
                        default='solution.robupy.pkl',
                        help='solution object')

    model, solution, simulate = _distribute_inputs(parser)

    solve(model, solution, simulate)