#!/usr/bin/env python
""" This module is used to create an indifference curve that outlines the
modeling trade-offs.
"""
# module-wide parameters
NUM_PROCS = 5

AMBIGUITY_GRID = [0.00, 0.01, 0.02]

COST_GRID = dict()
COST_GRID[0.00] = [9.21, 9.22, 9.23, 9.24, 9.25]
COST_GRID[0.01] = [9.21, 9.22, 9.23, 9.24, 9.25]
COST_GRID[0.02] = [9.21, 9.22, 9.23, 9.24, 9.25]

################################################################################
# Setup
################################################################################
# standard library
from multiprocessing import Pool
from functools import partial

import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from auxiliary import *

# Import function to that a fast version of the toolbox is available.
from modules.auxiliary import compile_package
compile_package('--fortran --optimization', True)

# Auxiliary objects
num_eval_points = len(COST_GRID[AMBIGUITY_GRID[0]])
num_ambi_points = len(AMBIGUITY_GRID)
################################################################################
# Checks that also help me understand how the module works.
################################################################################
# Check that points for all levels of ambiguity levels are defined
assert (set(COST_GRID.keys()) == set(AMBIGUITY_GRID))
# Check that the same number of points is requested. This ensures the
# symmetry of the evaluation grid for the parallelization request.
for key_ in AMBIGUITY_GRID:
    assert (len(COST_GRID[key_]) == num_eval_points)
# Make sure that there are no duplicates in the grid.
for key_ in AMBIGUITY_GRID:
    assert (len(COST_GRID[key_]) == len(set(COST_GRID[key_])))
################################################################################
# Main program
################################################################################
#  Starting with a clean slate
cleanup()
# Determine the baseline distribution
base_choices = get_baseline()
# Create the grid of the tasks. This collapses the hierarchical parallelism
# into one level.
tasks = []
for ambi in AMBIGUITY_GRID:
    for point in COST_GRID[key_]:
        tasks += [(ambi, point)]
# Prepare the function for multiprocessing by modifying interface.
criterion_function = partial(criterion_function, base_choices)
# Run multiprocessing module
p = Pool(NUM_PROCS)
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
# Write the information to file for visual inspection for now.
write_logging(AMBIGUITY_GRID, COST_GRID, final)

# TODO:  LOGGING, ACROPOLIS, ... CLEANUP AND BREAK GRAPH