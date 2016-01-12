#!/usr/bin/env python
""" This script solves and simulates the first RESTUD economy for a variety
of alternative degrees of ambiguity.
"""

# standard library
from multiprocessing import Pool
from functools import partial

import pickle as pkl
import numpy as np

import argparse
import socket
import shutil
import sys
import glob
import os

# Check for Python 3
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# project library
from modules.auxiliary import compile_package
from robupy import read

# Auxiliary functions
from auxiliary import solve_ambiguous_economy
from auxiliary import track_schooling_over_time
from auxiliary import track_final_choices

''' Functions
'''


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    num_procs, grid = args.num_procs,  args.grid
    is_recompile = args.is_recompile
    is_debug = args.is_debug

    # Check arguments
    assert (num_procs > 0)

    if grid != 0:
        assert (len(grid) == 3)
        assert (grid[2] > 0.0)
        assert (grid[0] < grid[1]) or ((grid[0] == 0) and (grid[1] == 0))

    # Finishing
    return num_procs, grid, is_recompile, is_debug


def create_results(init_dict, num_procs, grid):
    """ Solve the RESTUD economies for different levels of ambiguity.
    """
    # Construct grid
    if grid == 0:
        grid = [0.0]
    else:
        grid = np.linspace(start=grid[0], stop=grid[1], num=int(grid[2]))

    # Solve numerous economies
    process_tasks = partial(solve_ambiguous_economy, init_dict, is_debug)
    p = Pool(num_procs)
    p.map(process_tasks, grid)


    # Cleanup
    for file_ in glob.glob('*.log'):
        os.remove(file_)


def process_results(init_dict, is_debug):
    """ Process results from the models.
    """

    track_final_choices(init_dict, is_debug)

    track_schooling_over_time()


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Solve ROBUST economy for varying level of ambiguity.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
        default=1, help='use multiple processors')

    parser.add_argument('--grid', action='store', type=float, dest='grid',
        default=[0, 0.1, 2], nargs='+',
        help='construct grid using np.linspace (start, stop, num)')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only three periods')

    # Process command line arguments
    num_procs, grid, is_recompile, is_debug = distribute_arguments(
        parser)

    # Start with a clean slate.
    os.system('./clean')

    # Read the baseline specification and obtain the initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')
    init_dict = read('model.robupy.ini').get_attr('init_dict')
    os.unlink('model.robupy.ini')

    # For a smooth graph, we increase the number of simulated agents to 10,000.
    init_dict['SIMULATION']['agents'] = 10000

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if is_recompile:
        if 'acropolis' in socket.gethostname():
            compile_package('--fortran', True)
        else:
            compile_package('--fortran --debug', True)

    # Run tasks
    create_results(init_dict, num_procs, grid)

    # Process results
    os.mkdir('rslts')
    process_results(init_dict, is_debug)
