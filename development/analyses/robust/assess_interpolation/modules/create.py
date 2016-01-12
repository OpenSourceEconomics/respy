#!/usr/bin/env python
""" This module assess the quality of interpolation.
"""

# standard library
from multiprocessing import Pool

import argparse
import shutil
import socket
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from auxiliary import distribute_arguments
from auxiliary import aggregate_results
from auxiliary import process_tasks

from modules.auxiliary import compile_package

# module-wide variable
SPECIFICATIONS = ['one', 'two', 'three']
SOLUTIONS = ['full', 'approximate']

''' Core function
'''
def run(level, num_procs):

    # Auxiliary objects
    name = '%03.3f' % level

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if 'acropolis' in socket.gethostname():
        compile_package('--fortran', True)
    else:
        compile_package('--fortran --debug', True)

    # Create directory structure. The upper directory is denoted by the level
    # of ambiguity. Below, we distinguish between the full and approximate
    # solution approach. Then different directories are generated for all
    # three specifications.
    if os.path.exists(name):
        shutil.rmtree(name)

    os.mkdir(name), os.chdir(name)
    for solution in SOLUTIONS:
        for spec in SPECIFICATIONS:
            os.makedirs(solution + '/' + 'data_' + spec)

    # Set up tasks by combining all specifications with a full and approximate
    # solution request. Run all using multiprocessing capabilities.
    tasks = []
    for spec in SPECIFICATIONS:
        for solution in SOLUTIONS:
            tasks += [(spec, solution, level)]

    # Initialize the pool of multiple processors.
    p = Pool(num_procs)
    p.map(process_tasks, tasks)

    # Aggregate results across the different subdirectories where the full and
    # approximate solutions are stored.
    aggregate_results(SPECIFICATIONS, SOLUTIONS)


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

    level, num_procs = distribute_arguments(parser)

    run(level, num_procs)
