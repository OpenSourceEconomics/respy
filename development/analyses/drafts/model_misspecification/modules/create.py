#!/usr/bin/env python
""" This module allows to assess the implications of model misspecification
for estimates of psychic costs. We generate a simulated sample with some
level of ambiguity and then fit a risk-only model.
"""

# standard library
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import partial

import pickle as pkl

import argparse
import socket
import shutil
import glob
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# project library
from auxiliary import solve_estimated_economy
from auxiliary import distribute_arguments
from auxiliary import solve_true_economy
from auxiliary import criterion_function
from auxiliary import prepare_directory

from modules.auxiliary import compile_package

from robupy import read


''' Core function
'''


def run(init_dict, is_restart, is_debug, level):
    """ Inspect the implications of model misspecification for the estimates
    of psychic costs.
    """
    # Auxiliary objects
    name = '%03.3f' % level

    # Prepare the directory structure. We initialize a fresh directory named
    # by the level of ambiguity in the baseline economy. If restart, then the
    # existing directory is cleaned and just the information on the true
    # economy is retained. Finally, we move into the subdirectory.
    prepare_directory(is_restart, name)
    os.chdir(name)

    # Store the information about the true underlying data generating process in
    # a subdirectory and read in base choices. Or just read it in from a
    # previous run.
    if not is_restart:
        solve_true_economy(level, init_dict, is_debug)
    base_choices = pkl.load(open('true/base_choices.pkl', 'rb'))

    # Modification from baseline initialization file. This is not really
    # necessary but improves readability.
    init_dict['AMBIGUITY']['level'] = 0.00

    # Criterion function uses update. We optimize over the intercept in the
    # reward function.
    x0 = init_dict['EDUCATION']['int']
    opt = minimize(criterion_function, x0, args=(base_choices, init_dict,
                    is_debug), method="Nelder-Mead")

    # Write out some basic information to a file.
    with open('misspecification.robupy.log', 'a') as file_:

        file_.write('    Model Misspecification \n')
        file_.write('    ---------------------- \n\n')

        string = '    {0[0]:<5} {0[1]:7.4f}\n\n'
        file_.write(string.format(['Result', opt['x'][0]]))

        file_.write('    Function ' + str(opt['fun']) + '\n')
        file_.write('    Success  ' + str(opt['success']) + '\n')
        file_.write('    Message  ' + opt['message'] + '\n\n')

    # Solve the estimated economy to compare
    interpect = solve_estimated_economy(opt, init_dict, is_debug)

    # Cleanup scratch files generated during estimation. Other useful
    # material is retained in subdirectories.
    for file_ in glob.glob('*.robupy.*'):
        # Keep information about optimization
        if 'misspecification' in file_:
            continue
        os.unlink(file_)

    os.chdir('../')

    # Finishing
    return interpect

''' Execution of module as script.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--levels', action='store', type=float, dest='levels',
        required=True, nargs='+', help='level of ambiguity in true economy')

    parser.add_argument('--restart', action='store_true', dest='is_restart',
        help='restart using existing results')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only five periods')

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
         default=1, help='use multiple processors')

    # Distribute attributes
    levels, num_procs, is_restart, is_recompile, is_debug = \
        distribute_arguments(parser)

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
    process_tasks = partial(run, init_dict, is_restart, is_debug)
    intercepts = Pool(num_procs).map(process_tasks, levels)

    # Restructure return arguments for better interpretability and further
    # processing.
    rslt = dict()
    for i, level in enumerate(levels):
        rslt[level] = intercepts[i]

    # Store for further processing
    os.mkdir('rslts')
    pkl.dump(rslt, open('rslts/model_misspecification.robupy.pkl', 'wb'))