#!/usr/bin/env python
""" This script solves and simulates the first RESTUD economy for a variety
of alternative degrees of ambiguity.
"""

# standard library
from multiprocessing import Pool

import pickle as pkl
import numpy as np

import argparse
import sys
import glob
import os

# Check for Python 3
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')

# Auxiliary functions
from auxiliary import plot_schooling_ambiguity
from auxiliary import plot_choices_ambiguity

from auxiliary import track_schooling_over_time
from auxiliary import track_final_choices

from auxiliary import solve_ambiguous_economy
from auxiliary import get_levels

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')

# project library
from modules.auxiliary import compile_package


''' Functions
'''


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    num_procs, grid, is_graphs = args.num_procs,  args.grid, args.graphs

    # Check arguments
    assert (num_procs > 0)

    if grid != 0:
        assert (len(grid) == 3)
        assert (grid[2] > 0.0)
        assert (grid[0] < grid[1]) or ((grid[0] == 0) and (grid[1] == 0))

    # Finishing
    return num_procs, grid, is_graphs


def create_results(num_procs, grid):
    """ Solve the RESTUD economies for different levels of ambiguity.
    """
    # Cleanup
    os.system('./clean')

    # Construct grid
    if grid == 0:
        grid = [0.0]
    else:
        grid = np.linspace(start=grid[0], stop=grid[1], num=int(grid[2]))

    # Prepare directory
    os.mkdir('rslts')

    os.chdir('rslts')

    # Solve numerous economies
    p = Pool(num_procs)
    p.map(solve_ambiguous_economy, grid)

    os.chdir('../')

    # Cleanup
    for file_ in glob.glob('*.log'):
        os.remove(file_)


def process_results():
    """ Process results from the models.
    """

    track_final_choices()

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

    parser.add_argument('--graphs', action='store_true', dest='graphs',
        default=False, help='create only graphs')

    # Process command line arguments
    num_procs, grid, is_graphs = distribute_arguments(parser)

    # Run tasks
    if not is_graphs:

        compile_package('--fortran --optimization', True)

        create_results(num_procs, grid)

        process_results()

    # Plotting
    rslt = pkl.load(open('rslts/ambiguity_shares_final.pkl', 'rb'))

    # TODO: Remove fix
    #import numpy as np
    #rslt['levels'] = np.linspace(0, 0.02, 11)
    plot_choices_ambiguity(rslt)

    rslt = pkl.load(open('rslts/ambiguity_shares_time.pkl', 'rb'))
    plot_schooling_ambiguity(rslt)
