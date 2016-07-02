#!/usr/bin/env python
""" Script to run the whole vault of regression tests.
"""
# standard library
from __future__ import print_function

import pickle as pkl
import numpy as np

import subprocess
import sys
import os

# testing library
from auxiliary import read_request
from config import python2_exec
from config import python3_exec

# Reconstruct directory structure and edits to PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/testing/regression/modules', '')

PYTHON_VERSION = sys.version_info[0]

if PYTHON_VERSION == 2:
    python_exec = python2_exec
else:
    python_exec = python3_exec

num_tests = read_request()

# We need to be explicit about the PYTHON version as otherwise the F2PY
# libraries are not compiled in accordance with the PYTHON version used by
# for the execution of the script.
cwd = os.getcwd()
os.chdir(PACKAGE_DIR + '/respy')
subprocess.check_call(python_exec + ' waf distclean', shell=True)
subprocess.check_call(python_exec + ' waf configure build --debug', shell=True)
os.chdir(cwd)

# Import package. The late import is required as the compilation needs to
# take place first.
from respy import RespyCls
from respy import simulate
from respy import estimate

sys.path.insert(0, PACKAGE_DIR + '/respy/tests')
from codes.random_init import generate_init

############################################################################
# RUN
############################################################################
fname = 'test_vault_' + str(PYTHON_VERSION) + '.respy.pkl'

tests = []
for idx in range(num_tests):
    print('\n Creating Test ', idx, 'with version ', PYTHON_VERSION)

    constr = dict()
    constr['maxfun'] = int(np.random.choice([0, 1, 2, 3, 5, 6], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
    constr['flag_scaling'] = np.random.choice([True, False], p=[0.1, 0.9])

    init_dict = generate_init(constr)

    respy_obj = RespyCls('test.respy.ini')

    simulate(respy_obj)

    crit_val = estimate(respy_obj)[1]

    test = (init_dict, crit_val)

    tests += [test]

    pkl.dump(tests, open(fname, 'wb'))
