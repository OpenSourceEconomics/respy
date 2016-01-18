#!/usr/bin/env python
""" This module allows to assess the implications of model misspecification
for estimates of psychic costs. We generate a simulated sample with some
level of ambiguity and then fit a risk-only model.
"""

# standard library
from multiprocessing import Pool
from functools import partial

import argparse
import socket
import shutil
import copy
import sys
import os

# scipy library
import numpy as np

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/analyses/robust/_scripts')
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# _scripts
from _auxiliary import float_to_string

# local library
from auxiliary import distribute_arguments
from auxiliary import solve_true_economy
from auxiliary import criterion_function

# testing library
from modules.auxiliary import compile_package

# robupy library
from robupy import read


''' Core function
'''


def run(base_dict, base_choices, is_debug, level, intercept):
    """ Inspect the implications of model misspecification for the estimates
    of psychic costs.
    """
    # Prepare directory structure.
    str_ = float_to_string(intercept)
    os.mkdir(str_), os.chdir(str_)

    # Modification from baseline initialization file. This is not really
    # necessary but improves readability.
    init_dict = copy.deepcopy(base_dict)
    init_dict['AMBIGUITY']['level'] = level
    init_dict['EDUCATION']['int'] = intercept

    if is_debug:
        init_dict['BASICS']['periods'] = 5

    # Evaluate criterion function at grid point.
    rslt = criterion_function(init_dict, base_choices)

    # Write out some basic information to a file.
    with open('indifference_curve.robupy.log', 'a') as file_:

        file_.write('    Indifference Curve \n')
        file_.write('    ------------------ \n\n')

        string = '    {0[0]:<15}     {0[1]:7.4f}\n\n'
        file_.write(string.format(['Intercept', intercept]))
        file_.write(string.format(['Level', level]))

        string = '    {0[0]:<15} {0[1]:17.10f}\n\n'
        file_.write(string.format(['Criterion', rslt]))

    # Finishing
    os.chdir('../')


''' Execution of module as script.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--level', action='store', type=float, dest='level',
        default=0.0, help='level of ambiguity in economy')

    parser.add_argument('--spec', action='store', type=str, dest='spec',
        default='one', help='baseline specification')

    parser.add_argument('--grid', action='store', type=float, dest='grid',
        default=[10, 20, 2], nargs=3,
        help='input for grid creation: lower, upper, points')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only five periods')

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
         default=1, help='use multiple processors')

    # Distribute attributes
    level, grid, num_procs, is_recompile, is_debug, spec = \
        distribute_arguments(parser)

    # Prepare directory structure.
    try:
        os.mkdir('rslts')
    except FileExistsError:
        pass

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if is_recompile:
        if 'acropolis' in socket.gethostname():
            compile_package('--fortran', True)
        else:
            compile_package('--fortran --debug', True)

    # Read the baseline specification and obtain the initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_' + spec + '.robupy.ini', 'model.robupy.ini')
    base_dict = read('model.robupy.ini').get_attr('init_dict')
    os.unlink('model.robupy.ini')

    # Create and switch directory. Then solve the true economy for the
    # baseline distribution.
    base_dict['SIMULATION']['agents'] = 10000

    str_ = float_to_string(level)
    os.chdir('rslts'), os.mkdir(str_), os.chdir(str_)
    base_choices = solve_true_economy(base_dict, is_debug)

    # Here I need the grid creation for the mp processing.
    start, end, num = grid
    intercepts = np.linspace(start, end, num)

    # Create pool of processors and send off requests.
    process_tasks = partial(run, base_dict, base_choices, is_debug, level)
    Pool(num_procs).map(process_tasks, intercepts)

    # Aggregate results. This include results from other existing runs.
    os.chdir('../../'), os.system('./aggregate')