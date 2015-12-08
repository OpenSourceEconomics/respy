#!/usr/bin/env python
""" This module is used to create an indifference curve that outlines the
modeling tradeoffs.
"""

# standard library
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from functools import partial

import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from modules.auxiliary import compile_package
from auxiliary import *

# Import function to that a fast version of the toolbox is available.
#compile_package('--fortran --optimization', True)

# module-wide parameters
OUTER_NUM_PROCS = 3
INNER_NUM_PROCS = 2

AMBIGUITY_GRID = [0.00, 0.01, 0.02]

COST_GRID = dict()
COST_GRID[0.00] = [9.21, 9.22, 9.23, 9.24, 9.25]
COST_GRID[0.01] = [9.21, 9.22, 9.23, 9.24, 9.25]
COST_GRID[0.02] = [9.21, 9.22, 9.23, 9.24, 9.25]

# Starting with a clean slate
cleanup()

# Determine the baseline distribution
base_choices = get_baseline()

# Create a multiprocessing pool to solve for the required information for
# each level of ambiguity at once.

# HIERARCHINCAL PARALLELISM

#p = Pool(OUTER_NUM_PROCS)

#distributed_function = partial(solve_ambiguous_economy, base_choices,
#                                COST_GRID, INNER_NUM_PROCS)

#p.map(distributed_function, AMBIGUITY_GRID)

# TODO: PARALLELIZATION, ACROPOLIS, GRAPH

