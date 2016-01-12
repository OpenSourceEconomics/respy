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

from robupy.tests.random_init import print_random_dict

from robupy import read
from robupy import solve

# Auxiliary functions
from auxiliary import distribute_arguments
from auxiliary import get_robupy_obj
from auxiliary import get_results

''' Functions
'''


def run(init_dict, is_debug, level):
    """ Solve an economy in a subdirectory.
    """
    # Formatting directory name
    name = '{0:0.3f}'.format(level)

    # Create directory structure.
    os.mkdir(name), os.chdir(name)

    # Update level of ambiguity
    init_dict['AMBIGUITY']['level'] = level

    # For debugging purposes, write out initialization file.
    print_random_dict(init_dict)

    # Solve the basic economy
    solve(get_robupy_obj(init_dict))

    # Finishing
    os.chdir('../')


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
    num_procs, grid, is_recompile, is_debug = distribute_arguments(parser)

    # Start with a clean slate.
    os.system('./clean')

    # Read the baseline specification and obtain the initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')
    base_dict = read('model.robupy.ini').get_attr('init_dict')
    os.unlink('model.robupy.ini')

    # For a smooth graph, we increase the number of simulated agents to 10,000.
    base_dict['SIMULATION']['agents'] = 10000

    if is_debug:
        base_dict['BASICS']['periods'] = 3
        base_dict['SIMULATION']['agents'] = 100

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if is_recompile:
        if 'acropolis' in socket.gethostname():
            compile_package('--fortran', True)
        else:
            compile_package('--fortran --debug', True)

    # Construct a grid for the evaluation.
    start, stop, num = grid
    grid = np.linspace(start, stop, num)

    # Solve numerous economies
    process_tasks = partial(run, base_dict, is_debug)
    Pool(num_procs).map(process_tasks, grid)

    # Construct results from the results file.
    rslts = get_results(base_dict, is_debug)

    # Process results
    os.mkdir('rslts')
    pkl.dump(rslts, open('rslts/ambiguity_choices.robupy.pkl', 'wb'))
