#!/usr/bin/env python
""" This module allows to investigate the policy impact under
different levels of ambiguity and subsidy.
"""

# standard library
from multiprocessing import Pool
from functools import partial

import pickle as pkl
import argparse
import shutil
import socket
import shlex
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# Auxiliary functions
from modules.auxiliary import compile_package
from auxiliary import get_robupy_obj
from auxiliary import get_name

from robupy import read
from robupy import solve

''' Auxiliary functions
'''


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    is_recompile = args.is_recompile
    num_procs = args.num_procs
    is_debug = args.is_debug

    # Check arguments
    assert (num_procs > 0)
    assert (is_debug in [True, False])
    assert (is_recompile in [True, False])

    # Finishing
    return num_procs, is_recompile, is_debug


def process_models(args):
    """ This function processes the information from all simulated models.
    """

    # Prepare the resulting dictionary.
    rslt = dict()
    for arg in args:
        # Distribute elements of request
        _, level, subsidy = arg
        # Create required keys in dictionary
        if level not in rslt.keys():
            rslt[level] = dict()
        # Add levels
        rslt[level][subsidy] = []

    # Collect all results from the subdirectories.
    for arg in args:
        # Distribute elements of request
        _, level, subsidy = arg
        # Auxiliary objects
        name = get_name(level, subsidy)
        # Switch to results
        os.chdir(name)
        # Extract all results
        with open('data.robupy.info', 'r') as rslt_file:
            # Store results
            for line in rslt_file:
                list_ = shlex.split(line)
                try:
                    if 'Education' == list_[1]:
                        rslt[level][subsidy] += [float(list_[2])]
                except IndexError:
                    pass
        # Return to root directory.
        os.chdir('../../')

    # Finishing
    return rslt


def run(init_dict, is_debug, args):
    """ This function solves and simulates all requested models.
    """
    # Distribute input arguments
    intercept, level, subsidy = args

    # Switch to subdirectory, create if required.
    name = get_name(level, subsidy)
    if not os.path.exists(name):
        os.makedirs(name)
    os.chdir(name)

    # Construct request
    init_dict['AMBIGUITY']['level'] = level
    init_dict['EDUCATION']['coeff'][1] -= subsidy
    init_dict['EDUCATION']['int'] = intercept

    if is_debug:
        init_dict['BASICS']['periods'] = 3

    # Solve the basic economy
    solve(get_robupy_obj(init_dict))

    # Return to root directory. This is useful when running only a scalar
    # process during debugging.
    os.chdir('../../')

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create material for ROBUST lecture about policy '
        'effects.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
        default=1, help='use multiple processors')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only five periods')

    # In the first iteration of the code, I use the hardcoded results from
    # the independent model misspecification exercise and the policy schedule.
    LEVELS, INTERCEPTS = [0.00, 0.01, 0.02], [0.00, -3000.00, -5000.00]
    SUBSIDIES = [0.00, 500.00, 1000.00]

    # Process command line arguments
    num_procs, is_recompile, is_debug = distribute_arguments(parser)

    # Start with a clean slate.
    os.system('./clean')

    # Read the baseline specification and obtain the initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')
    init_dict = read('model.robupy.ini').get_attr('init_dict')
    os.unlink('model.robupy.ini')

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if is_recompile:
        if 'acropolis' in socket.gethostname():
            compile_package('--fortran', True)
        else:
            compile_package('--fortran --debug', True)

    # Construct all estimation requests for the policy analysis.
    args = []
    for i in range(3):
        intercept, level = INTERCEPTS[i], LEVELS[i]
        for j in range(3):
            subsidy = SUBSIDIES[j]
            args += [(intercept, level, subsidy)]

    # Set up pool for processors for parallel execution.
    process_tasks = partial(run, init_dict, is_debug)
    Pool(num_procs).map(process_tasks, args)

    # Aggregate results from all subdirectories created during the request.
    rslt = process_models(args)

    # Store results for further processing
    os.mkdir('rslts')
    pkl.dump(rslt, open('rslts/policy_impact.robupy.pkl', 'wb'))

