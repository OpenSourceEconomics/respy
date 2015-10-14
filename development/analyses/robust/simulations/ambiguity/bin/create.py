#!/usr/bin/env python
""" This script solves and simulates the first RESTUD economy for a variety
of alternative degrees of ambiguity.
"""

# standard library
from multiprocessing import Pool

import pickle as pkl
import numpy as np

import argparse
import shlex
import glob
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
from robupy import solve
from robupy import read

# module-wide variables
OCCUPATIONS = ['Occupation A', 'Occupation B', 'Schooling', 'Home']


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
        assert (grid[2] > 0.0)
        assert (grid[0] < grid[1]) or ((grid[0] == 0) and (grid[1] == 0))

    # Finishing
    return num_procs, grid


def solve_ambiguous_economy(level):
    """ Solve an economy in a subdirectory.
    """
    # Process baseline file
    robupy_obj = read('../model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')

    # Formatting directory name
    name = '{0:0.3f}'.format(level)

    # Create directory
    os.mkdir(name)

    os.chdir(name)

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


def get_levels():
    """ Infer ambiguity levels from directory structure.
    """
    os.chdir('rslts')

    levels = []

    for level in glob.glob('*/'):
        # Cleanup strings
        level = level.replace('/', '')
        # Collect levels
        levels += [level]

    os.chdir('../')

    # Finishing
    return sorted(levels)


def track_final_choices():
    """ Track the final choices from the ROBUPY output.
    """
    # Auxiliary objects
    levels = get_levels()

    # Process benchmark file
    robupy_obj = read('model.robupy.ini')

    num_periods = robupy_obj.get_attr('num_periods')

    # Create dictionary with the final shares for varying level of ambiguity.
    shares = dict()
    for occu in OCCUPATIONS:
        shares[occu] = []

    # Iterate over all available ambiguity levels
    for level in levels:
        file_name = 'rslts/' + level + '/data.robupy.info'
        with open(file_name, 'r') as output_file:
            for line in output_file.readlines():
                # Split lines
                list_ = shlex.split(line)
                # Skip empty lines
                if not list_:
                    continue
                # Extract shares
                if str(num_periods) in list_[0]:
                    for i, occu in enumerate(OCCUPATIONS):
                        shares[occu] += [float(list_[i + 1])]

    # Finishing
    pkl.dump(shares, open('rslts/ambiguity_shares_final.pkl', 'wb'))


def track_schooling_over_time():
    """ Create dictionary which contains the simulated shares over time for
    varying levels of ambiguity.
    """
    # Auxiliary objects
    levels = get_levels()

    # Iterate over results
    shares = dict()
    for level in levels:
        # Construct dictionary
        shares[level] = dict()
        for choice in OCCUPATIONS:
            shares[level][choice] = []
        # Process results
        file_name = 'rslts/' + level + '/data.robupy.info'
        with open(file_name, 'r') as output_file:
            for line in output_file.readlines():
                # Split lines
                list_ = shlex.split(line)
                # Check relevance
                try:
                    int(list_[0])
                except ValueError:
                    continue
                except IndexError:
                    continue
                # Process line
                for i, occu in enumerate(OCCUPATIONS):
                    shares[level][occu] += [float(list_[i + 1])]

    # Finishing
    pkl.dump(shares, open('rslts/ambiguity_shares_time.pkl', 'wb'))


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Solve ROBUST economy for varying level of ambiguity.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
        default=1, help='use multiple processors')

    parser.add_argument('--grid', action='store', type=float, dest='grid',
        default=[0, 0, 1], nargs='+',
        help='construct grid using np.linspace (start, stop, num)')

    # Process command line arguments
    num_procs, grid = distribute_arguments(parser)

    # Run tasks
    compile_package('--fortran --optimization', True)

    create(num_procs, grid)

    track_final_choices()

    track_schooling_over_time()