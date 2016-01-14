#!/usr/bin/env python
""" This module assess the quality of interpolation.
"""

# standard library
from multiprocessing import Pool
from functools import partial

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
from _auxiliary import float_to_string
from _auxiliary import get_robupy_obj

# local library
from auxiliary import distribute_arguments
from auxiliary import aggregate_results

# robupy library
from robupy.tests.random_init import print_random_dict
from robupy import solve
from robupy import read

# testing library
from modules.auxiliary import compile_package

# module-wide variable
SPECIFICATIONS = ['one', 'two', 'three']
SOLUTIONS = ['full', 'approximate']

''' Core function
'''


def run(is_debug, task):
    """ Process tasks to simulate and solve.
    """
    # Distribute requested task
    spec, solution, level = task

    # Enter appropriate subdirectory
    os.chdir(solution), os.chdir('data_' + spec)

    # Copy relevant initialization file
    src = SPEC_DIR + '/data_' + spec + '.robupy.ini'
    tgt = 'model.robupy.ini'

    shutil.copy(src, tgt)

    # Prepare initialization file
    robupy_obj = read('model.robupy.ini')
    init_dict = robupy_obj.get_attr('init_dict')

    # Modification from baseline initialization file
    init_dict['AMBIGUITY']['level'] = level
    init_dict['SOLUTION']['store'] = True
    init_dict['PROGRAM']['debug'] = True

    if is_debug:
        init_dict['BASICS']['periods'] = 3

    # Decide for interpolation or full solution
    is_full = (solution == 'full')
    if is_full:
        init_dict['INTERPOLATION']['apply'] = False
    else:
        init_dict['INTERPOLATION']['apply'] = True

    # Print out initialization file for debugging.
    print_random_dict(init_dict)

    # Solve the modified model.
    solve(get_robupy_obj(init_dict))

    os.chdir('../../')


''' Execution of module as script.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess the quality of interpolation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
         default=1, help='use multiple processors')

    parser.add_argument('--level', action='store', type=float, dest='level',
         default=0.00, help='level of ambiguity')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only three periods')

    # Distribute input
    level, num_procs, is_recompile, is_debug = distribute_arguments(parser)

    # Cleanup
    os.system('./clean')

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if is_recompile:
        if 'acropolis' in socket.gethostname():
            compile_package('--fortran', True)
        else:
            compile_package('--fortran --debug', True)

    # Auxiliary objects
    name = float_to_string(level)

    # Create directory structure. The upper directory is denoted by the level
    # of ambiguity. Below, we distinguish between the full and approximate
    # solution approach. Then different directories are generated for all
    # three specifications.
    if os.path.exists(name):
        shutil.rmtree(name)

    os.mkdir('rslts'), os.chdir('rslts'), os.mkdir(name), os.chdir(name)
    for solution in SOLUTIONS:
        for spec in SPECIFICATIONS:
            os.makedirs(solution + '/' + 'data_' + spec)

    # Set up tasks by combining all specifications with a full and approximate
    # solution request. Run all using multiprocessing capabilities.
    tasks = []
    for spec in SPECIFICATIONS:
        for solution in SOLUTIONS:
            tasks += [(spec, solution, level)]

    # Solve numerous economies
    process_tasks = partial(run, is_debug)
    Pool(num_procs).map(process_tasks, tasks)

    # Process results
    aggregate_results(SPECIFICATIONS, SOLUTIONS)
