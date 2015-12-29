#!/usr/bin/env python
""" This module is used to create an indifference curve that outlines the
modeling trade-offs between psychic costs and ambiguity.
"""

# standard library
from multiprocessing import Pool
from functools import partial

import numpy as np

import argparse
import socket
import shutil
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# project library
from auxiliary import distribute_arguments
from auxiliary import pair_evaluation
from auxiliary import write_logging
from auxiliary import get_baseline
from auxiliary import check_grid

from clean import cleanup
from robupy import read

# Make sure that fast evaluation of the model is possible
from modules.auxiliary import compile_package

''' Main functions
'''


def grid_search(AMBIGUITY_GRID, COST_GRID, num_procs, is_debug):
    """ Perform a grid search with all possible combinations.
    """
    # Check input
    check_grid(AMBIGUITY_GRID, COST_GRID)
    # Auxiliary objects
    num_eval_points = len(COST_GRID[AMBIGUITY_GRID[0]])
    num_ambi_points = len(AMBIGUITY_GRID)
    #  Starting with a clean slate
    cleanup()
    # Process baseline initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')
    init_dict = read('model.robupy.ini').get_attr('init_dict')
    # For debugging purposes, the number of periods is set to three.
    if is_debug:
        init_dict['BASICS']['periods'] = 3
    # Determine the baseline distribution
    base_choices = get_baseline(init_dict)
    # Create the grid of the tasks. This collapses the hierarchical parallelism
    # into one level.
    tasks = []
    for ambi in AMBIGUITY_GRID:
        for point in COST_GRID[ambi]:
            tasks += [(ambi, point)]
    # Prepare the function for multiprocessing by modifying interface.
    criterion_function = partial(pair_evaluation, init_dict, base_choices)
    # Run multiprocessing module
    p = Pool(num_procs)
    rslts = p.map(criterion_function, tasks)
    # Mapping the results from each evaluation back to an interpretable array.
    # The first dimension corresponds to the level of ambiguity while the second
    # dimension refers to the evaluation of the other point.
    final = np.empty((num_ambi_points, num_eval_points))
    for i, ambi in enumerate(AMBIGUITY_GRID):
        for j, point in enumerate(COST_GRID[ambi]):
            for k, task in enumerate(tasks):
                if ambi != task[0] or point != task[1]:
                    continue
                final[i, j] = rslts[k]
    # Finishing
    return final


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create indifference curve to illustrate modeling '
        'trade-offs.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
         default=1, help='use multiple processors')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only three periods')

    # Process command line arguments
    num_procs, is_debug = distribute_arguments(parser)

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if 'acropolis' in socket.gethostname():
        compile_package('--fortran', True)
    else:
        compile_package('--fortran --debug', True)

    ############################################################################
    # Manual parametrization of grid search.
    ############################################################################
    if is_debug:

        AMBIGUITY_GRID = [0.00, 0.01]

        COST_GRID = dict()
        COST_GRID[0.00] = np.linspace(0, -10000, num=1)
        COST_GRID[0.01] = np.linspace(0, -10000, num=1)

    else:

        AMBIGUITY_GRID = [0.00, 0.01, 0.02, 0.03]

        COST_GRID = dict()
        COST_GRID[0.00] = np.linspace(0, -10000, num=21)
        COST_GRID[0.01] = np.linspace(0, -10000, num=21)
        COST_GRID[0.02] = np.linspace(0, -10000, num=21)
        COST_GRID[0.03] = np.linspace(0, -10000, num=21)

    ############################################################################
    ############################################################################

    # Evaluate points on grid
    evals = grid_search(AMBIGUITY_GRID, COST_GRID, num_procs, is_debug)
    # Write the information to file for visual inspection for now.
    write_logging(AMBIGUITY_GRID, COST_GRID, evals)
    # Cleanup intermediate files, but keep the output with results.
    cleanup(False)
