#!/usr/bin/env python
""" This script attempts to express the uncertainty quantification.
"""

# standard library
from multiprocessing import Pool
from functools import partial

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
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# project library
from robupy import solve
from robupy import read

from modules.auxiliary import compile_package

from auxiliary import distribute_arguments
from auxiliary import get_robupy_obj


def run(is_debug, ambiguity_level):
    """ Extract expected lifetime value.
    """
    # Auxiliary objects
    name = '%03.3f' % ambiguity_level

    # Prepare directory structure
    os.mkdir(name), os.chdir(name)

    # Initialize auxiliary objects
    init_dict['AMBIGUITY']['level'] = ambiguity_level

    # Restrict number of periods for debugging purposes.
    if is_debug:
        init_dict['BASICS']['periods'] = 3

    # Solve the basic economy
    robupy_obj = solve(get_robupy_obj(init_dict))

    # Get the EMAX for further processing and extract relevant information.
    total_value = robupy_obj.get_attr('periods_emax')[0, 0]

    # Back to root directory
    os.chdir('../')

    # Finishing
    return total_value


''' Execution of module as script.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--levels', action='store', type=float, dest='levels',
        required=True, nargs='+', help='level of ambiguity in true economy')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only three periods')

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
         default=1, help='use multiple processors')

    # Cleanup
    os.system('./clean')

    # Distribute attributes
    levels, is_recompile, is_debug, num_procs = distribute_arguments(parser)

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

    # Set up pool for processors for parallel execution.
    process_tasks = partial(run, is_debug)
    rslts = Pool(num_procs).map(process_tasks, levels)

    # Restructure return arguments for better interpretability and further
    # processing. Add the baseline number.
    rslt = dict()
    for i, level in enumerate(levels):
        rslt[level] = rslts[i]

    # Store for further processing
    os.mkdir('rslts')
    pkl.dump(rslt, open('rslts/uncertainty_quantification.robupy.pkl', 'wb'))

