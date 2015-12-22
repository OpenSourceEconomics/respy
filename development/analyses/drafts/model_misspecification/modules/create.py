#!/usr/bin/env python
""" This module allows to assess the implications of model misspecification
for estimates of psychic costs. We generate a simulated sample with some
level of ambiguity and then fit a risk-only model.
"""

# standard library
from scipy.optimize import minimize

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

from robupy.tests.random_init import print_random_dict
from modules.auxiliary import compile_package

from robupy import read


''' Core function
'''


def run(level):
    """ Inspect the implications of model misspecification for the estimates
    of psychic costs.
    """
    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if 'acropolis' in socket.gethostname():
        compile_package('--fortran', True)
    else:
        compile_package('--fortran --debug', True)

    # Create a new directory with the level of ambiguity which houses the
    # estimation process and copy a baseline initialization file.
    name = '%03.3f' % level
    if os.path.exists(name):
        shutil.rmtree(name)
    os.mkdir(name), os.chdir(name)
    shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')

    # Store the information about the true underlying data generating process in
    # a subdirectory.
    base_choices = solve_true_economy(level)

    # Prepare initialization file
    robupy_obj = read('model.robupy.ini')
    init_dict = robupy_obj.get_attr('init_dict')

    # Modification from baseline initialization file
    init_dict['AMBIGUITY']['level'] = 0.00
    # TODO: Remove later
    #init_dict['BASICS']['periods'] = 5

    # Finalize initialization file and solve model
    print_random_dict(init_dict)

    # Criterion function uses update. We optimize over the intercept in the
    # reward function.
    x0 = init_dict['EDUCATION']['int']
    opt = minimize(criterion_function, x0, args=(base_choices,),
                   method="Nelder-Mead")

    # Write out some basic information to a file.
    with open('misspecification.robupy.log', 'a') as file_:

        file_.write('    Model Misspecification \n')
        file_.write('    ---------------------- \n\n')

        string = '    {0[0]:<5} {0[1]:7.4f}\n\n'
        file_.write(string.format(['Result', opt['x'][0]]))

        file_.write('    Success ' + str(opt['success']) + '\n')
        file_.write('    Message ' + opt['message'] + '\n\n')

    # Solve the estimated economy to compare
    solve_estimated_economy(opt)

    # Cleanup files generated during estimation
    for file_ in glob.glob('*.robupy.*'):
        # Keep information about optimization
        if 'misspecification' in file_:
            continue
        os.unlink(file_)


''' Execution of module as script.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--level', action='store', type=float, dest='level',
         default=0.00, help='level of ambiguity in true economy')

    level = distribute_arguments(parser)

    run(level)