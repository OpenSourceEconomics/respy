#!/usr/bin/env python
""" This module is used to create an indifference curve that outlines the
modeling trade-offs.
"""

# standard library
from multiprocessing import Pool
from functools import partial

import numpy as np

import argparse
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from auxiliary import pair_evaluation
from auxiliary import write_logging
from auxiliary import get_baseline
from clean import cleanup

# Make sure that fast evaluation of the model is possible
from modules.auxiliary import compile_package
compile_package('--fortran --optimization', True)

''' Main functions
'''


def grid_search(AMBIGUITY_GRID, COST_GRID, num_procs):
    """ Perform a grid search with all possible combinations.
    """
    # Auxiliary objects
    num_eval_points = len(COST_GRID[AMBIGUITY_GRID[0]])
    num_ambi_points = len(AMBIGUITY_GRID)
    #  Starting with a clean slate
    cleanup()
    # Determine the baseline distribution
    base_choices = get_baseline()
    # Create the grid of the tasks. This collapses the hierarchical parallelism
    # into one level.
    tasks = []
    for ambi in AMBIGUITY_GRID:
        for point in COST_GRID[ambi]:
            tasks += [(ambi, point)]
    # Prepare the function for multiprocessing by modifying interface.
    criterion_function = partial(pair_evaluation, base_choices)
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


def check_grid(AMBIGUITY_GRID, COST_GRID):
    """ Check the manual specification of the grid.
    """
    # Auxiliary objects
    num_eval_points = len(COST_GRID[0])
    # Check that points for all levels of ambiguity levels are defined
    assert (set(COST_GRID.keys()) == set(AMBIGUITY_GRID))
    # Check that the same number of points is requested. This ensures the
    # symmetry of the evaluation grid for the parallelization request.
    for key_ in AMBIGUITY_GRID:
        assert (len(COST_GRID[key_]) == num_eval_points)
    # Make sure that there are no duplicates in the grid.
    for key_ in AMBIGUITY_GRID:
        assert (len(COST_GRID[key_]) == len(set(COST_GRID[key_])))


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

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
         description='Create indifference curve to illustrate modeling '
                     'trade-offs.',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
         default=1, help='use multiple processors')

    ############################################################################
    # Manual parametrization of grid search.
    ############################################################################
    AMBIGUITY_GRID = [0.00, 0.01, 0.02]

    COST_GRID = dict()
    COST_GRID[0.00] = [9.21, 9.22, 9.23, 9.24, 9.25]
    COST_GRID[0.01] = [9.21, 9.22, 9.23, 9.24, 9.25]
    COST_GRID[0.02] = [9.21, 9.22, 9.23, 9.24, 9.25]
    ############################################################################
    ############################################################################

    # Process command line arguments
    num_procs = distribute_arguments(parser)
    # Evaluate points on grid
    evals = grid_search(AMBIGUITY_GRID, COST_GRID, num_procs)
    # Write the information to file for visual inspection for now.
    write_logging(AMBIGUITY_GRID, COST_GRID, evals)
    # Cleanup intermediate files, but keep the output with results.
    cleanup(False)

    # TODO: ... CLEANUP AND BREAK GRAPH

