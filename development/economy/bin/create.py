#!/usr/bin/env python
""" This script solves and simulates the first RESTUD economy for a variety
of alternative degrees of ambiguity.
"""

# standard library
from multiprocessing import Pool

import numpy as np

import argparse
import shutil
import glob
import sys
import os

# Check for Python 3
import sys
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# Import function to that a fast version of the toolbox is available.
from robupy.tests.random_init import print_random_dict
from modules.auxiliary import compile_package
from robupy import *

# Import clean
from clean import cleanup

''' Functions
'''


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    num_procs, grid = args.num_procs,  args.grid

    # Check arguments
    assert (num_procs > 0)

    if grid != 0:
        assert (len(grid) == 3)
        assert (all(isinstance(element, int) for element in grid))
        assert (grid[2] > 0.0)
        assert (grid[0] < grid[1]) or ((grid[0] == 0) and (grid[1] == 0))

    # Finishing
    return num_procs, grid


def solve_ambiguous_economy(level):
    """ Solve an economy in a subdirectory.
    """

    # Process baseline file
    robupy_obj = read('model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')

    # Create directory
    os.mkdir(str(level))

    # Solve
    os.chdir(str(level))

    # Update level of ambiguity
    init_dict['AMBIGUITY']['level'] = level

    # Write initialization file
    print_random_dict(init_dict)

    # Solve requested model
    robupy_obj = read('test.robupy.ini')

    solve(robupy_obj)

    # Finishing
    os.chdir('../')


def create(num_procs, grid):
    """ Solve the RESTUD economies for different levels of ambiguity.
    """
    # Cleanup
    cleanup()

    # Compile fast version of ROBUPY package
    compile_package('--fortran --optimization', True)

    # Construct grid
    if grid == 0:
        grid = [0.0]
    else:
        grid = np.linspace(start=grid[0], stop=grid[1], num=grid[2])

    # Solve numerous economies
    p = Pool(num_procs)
    p.map(solve_ambiguous_economy, grid)


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Solve RESTUD economy for varying level of ambiguity.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
        default=1, help='use multiple processors')

    parser.add_argument('--grid', action='store', type=int, dest='grid',
        default=[0, 0, 1], nargs='+',
        help='construct grid using np.linspace (start, stop, num)')

    num_procs, grid = distribute_arguments(parser)

    # Run function
    create(num_procs, grid)
