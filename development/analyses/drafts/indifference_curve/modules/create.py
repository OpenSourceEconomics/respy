#!/usr/bin/env python
""" This module is used to create an indifference curve that outlines the
modeling trade-offs between psychic costs and ambiguity.
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

from robupy import read

# Make sure that fast evaluation of the model is possible
from modules.auxiliary import compile_package


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

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    # Process command line arguments
    num_procs, is_recompile, is_debug = distribute_arguments(parser)

    # Cleanup
    os.system('./clean')

    # Read the baseline specification and obtain the initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')
    init_dict = read('model.robupy.ini').get_attr('init_dict')
    os.unlink('model.robupy.ini')

    # Get the baseline distribution of choices. For debugging purposes,
    # the number of periods can be set to three.
    if is_debug:
        init_dict['BASICS']['periods'] = 3
    base_choices = get_baseline(init_dict)

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if is_recompile:
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
        COST_GRID[0.00] = np.linspace(0, -10000, num=3)
        COST_GRID[0.01] = np.linspace(0, -10000, num=3)

    else:

        AMBIGUITY_GRID = [0.00, 0.01, 0.02, 0.03]

        COST_GRID = dict()
        COST_GRID[0.00] = np.linspace(0, -10000, num=21)
        COST_GRID[0.01] = np.linspace(0, -10000, num=21)
        COST_GRID[0.02] = np.linspace(0, -10000, num=21)
        COST_GRID[0.03] = np.linspace(0, -10000, num=21)

    ############################################################################
    ############################################################################

    # Create and process the grid of the tasks. This collapses the hierarchical
    # parallelism into one level.
    check_grid(AMBIGUITY_GRID, COST_GRID)
    tasks = []
    for ambi in AMBIGUITY_GRID:
        for point in COST_GRID[ambi]:
            tasks += [(ambi, point)]

    # Set up pool for processors for parallel execution.
    criterion_function = partial(pair_evaluation, init_dict, base_choices)
    rslts = Pool(num_procs).map(criterion_function, tasks)

    # Mapping the results from each evaluation back to an interpretable
    # dictionary. The key corresponds to the level of ambiguity and the
    # values are the results from the grid evaluation.
    final = dict()
    for ambi in AMBIGUITY_GRID:
        final[ambi] = []
        for j, point in enumerate(COST_GRID[ambi]):
            for k, task in enumerate(tasks):
                if ambi != task[0] or point != task[1]:
                    continue
                final[ambi] += [rslts[k]]

    # Store for further processing
    pkl.dump(final, open('indifference.robupy.pkl', 'wb'))

    # Write the information to file for visual inspection for now.
    write_logging(AMBIGUITY_GRID, COST_GRID, final)
