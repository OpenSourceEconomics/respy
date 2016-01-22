#!/usr/bin/env python
""" This module allows to investigate the policy impact under
different levels of ambiguity and subsidy.
"""

# standard library
from multiprocessing import Pool
from functools import partial
from copy import deepcopy

import pickle as pkl

import argparse
import shutil
import socket
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/analyses/robust/_scripts')
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# _scripts
from _auxiliary import get_indifference_points
from _auxiliary import get_robupy_obj

# testing library
from modules.auxiliary import compile_package

# local library
from auxiliary import distribute_arguments
from auxiliary import process_models
from auxiliary import get_name

# robupy library
from robupy.tests.random_init import print_random_dict
from robupy import read
from robupy import solve

''' Auxiliary functions
'''


def run(baseline_dict, is_debug, args):
    """ This function solves and simulates all requested models.
    """
    # Switch to subdirectory
    os.chdir('rslts')

    # Distribute input arguments
    intercept, level, subsidy = args

    # Switch to subdirectory, create if required.
    name = get_name(level, subsidy)
    if not os.path.exists(name):
        os.makedirs(name)
    os.chdir(name)

    # A deep copy is required other the subsidy increments accumulate.
    init_dict = deepcopy(baseline_dict)

    # Construct request
    init_dict['SIMULATION']['agents'] = 10000
    init_dict['AMBIGUITY']['level'] = level
    init_dict['EDUCATION']['coeff'][0] += subsidy
    init_dict['EDUCATION']['int'] = intercept

    if is_debug:
        init_dict['BASICS']['periods'] = 3

    # Store some additional information for later debugging.
    print_random_dict(init_dict)

    # Solve the basic economy
    solve(get_robupy_obj(init_dict))

    # Return to root directory. This is useful when running only a scalar
    # process during debugging.
    os.chdir('../../../')

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create material for ROBUST lecture about policy '
        'effects.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--spec', action='store', type=str, dest='spec',
        default='one', help='baseline specification')

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
        default=1, help='use multiple processors')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only five periods')

    # Hard-coded subsidy schedule
    SUBSIDIES = [0.00, 500.00, 1000.00, 1500.00, 2000.00]

    # Process command line arguments
    num_procs, is_recompile, is_debug, spec = distribute_arguments(parser)

    # Start with a clean slate.
    os.system('./clean'), os.mkdir('rslts')

    # In the first iteration of the code, I use the hardcoded results from
    # the independent model misspecification exercise and the policy schedule.
    if not is_debug:
        specifications = get_indifference_points()

    else:
        specifications = [(0.00, 0.00), (0.00, 0.00), (0.00, 0.00)]

    # Read the baseline specification and obtain the initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_' + spec + '.robupy.ini', 'model.robupy.ini')
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
    for i, specification in enumerate(specifications):
        level, intercept = specification
        for j, subsidy in enumerate(SUBSIDIES):
            args += [(intercept, level, subsidy)]

    # Set up pool for processors for parallel execution.
    process_tasks = partial(run, init_dict, is_debug)
    Pool(num_procs).map(process_tasks, args)

    # Aggregate results from all subdirectories created during the request.
    rslt = process_models(args)

    # Store results for further processing
    pkl.dump(rslt, open('rslts/policy_intervention.robupy.pkl', 'wb'))

