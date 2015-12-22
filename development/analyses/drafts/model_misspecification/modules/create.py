#!/usr/bin/env python
""" This module allows to assess the implications of model misspecification
for estimates of psychic costs.
"""

# standard library
from scipy.optimize import minimize

import socket
import shutil
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from auxiliary import solve_estimated_economy
from auxiliary import solve_true_economy
from auxiliary import criterion_function

from robupy.tests.random_init import print_random_dict
from modules.auxiliary import compile_package

from robupy import read

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'


def run(level):
    """ Determine
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

    # Finalize initialization file and solve model
    print_random_dict(init_dict)

    # Criterion function uses update. We optimize over the intercept in the
    # reward function.
    x0 = init_dict['EDUCATION']['int']
    opt = minimize(criterion_function, x0, args=(base_choices,),
                   method="Nelder-Mead")

    # Solve the estimated economy to compare
    solve_estimated_economy(opt)

# cleanup ... integreate in one fulnction with distinctio between before and
# after, depends on wheterh to delte true and estimated.

# Need some basic logging, which writes out difference in estimates in
# addition to optimization report.

level = 0.01
run(level)