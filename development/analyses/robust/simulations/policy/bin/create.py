#!/usr/bin/env python
""" This module allows to investigate the policy responsiveness under
different levels of ambiguity.
"""

# standard library
from multiprocessing import Pool
from functools import partial

import pickle as pkl
import numpy as np

import argparse
import shlex
import glob
import sys
import os

# Auxiliary functions
from auxiliary import formatting

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# Import function to that a fast version of the toolbox is available.
from robupy.tests.random_init import print_random_dict
from modules.auxiliary import compile_package

from robupy import read
from robupy import solve


# module-wide variables
AMBIGUITY_LEVELS = [0.00, 0.01, 0.02]

''' Auxiliary functions
'''


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    num_procs, num_points = args.num_procs,  args.num_points

    # Check arguments
    assert (num_procs > 0)
    assert (num_points > 0)

    # Finishing
    return num_procs, num_points


def process_models(num_points):
    """ This function processes the information from all simulated models.
    """

    # Switch to results
    os.chdir('rslts')

    # Auxiliary objects
    tuition_schedule = np.linspace(0, 4000, num_points)

    # Extract all results
    rslt = dict()
    for level in AMBIGUITY_LEVELS:
        os.chdir(formatting(level))

        rslt[level] = []

        for tuition in tuition_schedule:
            os.chdir(formatting(tuition))

            with open('data.robupy.info', 'r') as rslt_file:

                for line in rslt_file:
                    list_ = shlex.split(line)
                    try:
                        if 'Education' == list_[1]:
                            rslt[level] += [float(list_[2])]
                    except IndexError:
                        pass

            os.chdir('../')
        os.chdir('../')
    os.chdir('../')

    # Store for future processing
    pkl.dump(rslt, open('rslts/policy_responsiveness.pkl', 'wb'))


def mp_solve_model(init_dict, tuition):
    """ Solve a single model for a given level of tuition
    """

    # Create and change to subdirectory
    dir_ = formatting(tuition)
    os.mkdir(dir_), os.chdir(dir_)

    # Modify initialization file and write out
    init_dict['EDUCATION']['coeff'][1] = -tuition
    print_random_dict(init_dict)

    # Solve task
    robupy_obj = read('test.robupy.ini')
    solve(robupy_obj)

    os.chdir('../')


def solve_models(num_procs, num_points):
    """ This function solves and simulates all requested models.
    """
    # Cleanup
    os.system('./clean')

    # Create directory structure
    os.mkdir('rslts'), os.chdir('rslts')

    # Auxiliary objects
    tuition_schedule = np.linspace(0, 4000, num_points)

    # Read baseline specification
    robupy_obj = read('../model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')

    for level in AMBIGUITY_LEVELS:

        dir_ = formatting(level)
        os.mkdir(dir_), os.chdir(dir_)

        init_dict['AMBIGUITY']['level'] = level

        p = Pool(num_procs)
        p.map(partial(mp_solve_model, init_dict), tuition_schedule)

        os.chdir('../')
    os.chdir('../')

    # Cleanup
    for file_ in glob.glob('*.log'):
        os.remove(file_)


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create material for ROBUST lecture about policy '
        'responsiveness.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
        default=1, help='use multiple processors')

    parser.add_argument('--points', action='store', type=int, dest='num_points',
        default=1, help='number of evaluation points')

    # Process command line arguments
    num_procs, num_points = distribute_arguments(parser)

    # TODO: Comment back in, Update script, move formattting to auxiliary
    # script --> change in update
    #compile_package('--fortran --optimization', True)

    solve_models(num_procs, num_points)

    process_models(num_points)
