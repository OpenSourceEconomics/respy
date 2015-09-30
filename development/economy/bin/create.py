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

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/working/keane_economy/fort')
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
    num_procs = args.num_procs

    # Check arguments
    assert (num_procs > 0)

    # Finishing
    return num_procs


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

    import time

    start_time = time.time()

    solve(robupy_obj)

    print(time.time() - start_time)

    # Finishing
    os.chdir('../')


def create(num_procs):
    """ Solve the RESTUD economies for different levels of ambiguity.
    """
    # Cleanup
    cleanup()

    # Compile fast version of ROBUPY package
    compile_package('--fortran', True)

    # Solve numerous economies
    grid = [0.1]
    p = Pool(num_procs)
    p.map(solve_ambiguous_economy, grid)


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Solve RESTUD economy for varying level of ambiguity.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
                        default=1, help='number of available processors')

    num_procs = distribute_arguments(parser)

    # Run function
    create(num_procs)
