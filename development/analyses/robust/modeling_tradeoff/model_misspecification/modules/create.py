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
import copy
import glob
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
from _auxiliary import float_to_string

# robupy library
from robupy import read

# testing library
from modules.auxiliary import compile_package

# local library
from auxiliary import solve_estimated_economy
from auxiliary import distribute_arguments
from auxiliary import solve_true_economy
from auxiliary import criterion_function
from auxiliary import cleanup_directory
from auxiliary import SCALING

''' Core function
'''


def run(base_dict, is_debug, is_restart, args):
    """ Inspect the implications of model misspecification for the estimates
    of psychic costs.
    """
    # Switch to results directory.
    os.chdir('rslts')

    # Distribute arguments
    level, intercept = args

    # Auxiliary objects
    name = float_to_string(level)

    # Prepare the directory structure. We initialize a fresh directory named
    # by the level of ambiguity in the baseline economy. Finally, we move into
    # the subdirectory
    if not is_restart:
        os.mkdir(name), os.chdir(name)
    else:
        cleanup_directory(name)
        os.chdir(name)

    # Store the information about the true underlying data generating process in
    # a subdirectory and read in base choices. Or just read it in from a
    # previous run.
    init_dict = copy.deepcopy(base_dict)
    init_dict['AMBIGUITY']['level'] = level
    init_dict['EDUCATION']['int'] = intercept
    if is_debug:
        init_dict['BASICS']['periods'] = 5

    # Determine the baseline distribution of choices if required.
    if not is_restart:
        solve_true_economy(init_dict, is_debug)

    base_choices = pkl.load(open('true/base_choices.pkl', 'rb'))

    # Modification from baseline initialization file. This is not really
    # necessary but improves readability.
    init_dict['AMBIGUITY']['level'] = 0.00
    if is_debug:
        base_dict['BASICS']['periods'] = 5

    # Criterion function uses update. We optimize over the intercept in the
    # reward function.
    x0 = intercept/SCALING
    opt = minimize(criterion_function, x0, args=(base_choices, base_dict,
                     is_debug), method="Nelder-Mead")

    # Write out some basic information to a file.
    with open('model_misspecification.robupy.log', 'a') as file_:

        file_.write('    Model Misspecification \n')
        file_.write('    ---------------------- \n\n')

        string = '    {0[0]:<5} {0[1]:7.4f}\n\n'

        file_.write(string.format(['Result', opt['x'][0]*SCALING]))
        file_.write('    Function ' + str(opt['fun']) + '\n')
        file_.write('    Success  ' + str(opt['success']) + '\n')
        file_.write('    Message  ' + opt['message'] + '\n\n')

    # Solve the estimated economy to compare
    intercept = solve_estimated_economy(opt, base_dict)

    # Cleanup scratch files generated during estimation. Other useful
    # material is retained in subdirectories.
    for file_ in glob.glob('*.robupy.*'):
        # Keep information about optimization
        if 'misspecification' in file_:
            continue
        os.unlink(file_)

    os.chdir('../../')

    # Finishing
    return intercept

''' Execution of module as script.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--spec', action='store', type=str, dest='spec',
        default='one', help='baseline specification')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only five periods')

    parser.add_argument('--restart', action='store_true', dest='is_restart',
        help='restart estimation', default=False)

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
         default=1, help='use multiple processors')

    # Distribute attributes
    num_procs, is_recompile, is_debug, is_restart, spec = \
        distribute_arguments(parser)

    # Start with a clean slate
    if not is_restart:
        os.system('./clean'), os.mkdir('rslts')

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if is_recompile:
        if 'acropolis' in socket.gethostname():
            compile_package('--fortran', True)
        else:
            compile_package('--fortran --debug', True)

    # This exercise considers several economies that look rather identical to
    # the econometrican. However, they are generated by different combinations
    # of psychic cost and ambiguity. Each tuple (level, intercept) describes
    # such an economy. The values are determined by results from the
    # indifference curve exercise.
    if not is_debug:
        specifications = get_indifference_points()

    else:
        specifications = [(0.00, 0.00)]

    # If baseline choices are requested, then better check that these are in
    # fact available.
    if is_restart:
        for spec in specifications:
            level, _ = spec
            assert (os.path.exists('%03.3f' % level + '/true/base_choices.pkl'))

    # Read the baseline specification and obtain the initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_' + spec + '.robupy.ini', 'model.robupy.ini')
    base_dict = read('model.robupy.ini').get_attr('init_dict')
    os.unlink('model.robupy.ini')

    # Set up pool for processors for parallel execution.
    process_tasks = partial(run, base_dict, is_debug, is_restart)
    intercepts = Pool(num_procs).map(process_tasks, specifications)

    # Restructure return arguments for better interpretability and further
    # processing.
    rslt = dict()
    for i, args in enumerate(specifications):
        level, _ = args
        rslt[level] = intercepts[i]

    # Store for further processing
    pkl.dump(rslt, open('rslts/model_misspecification.robupy.pkl', 'wb'))