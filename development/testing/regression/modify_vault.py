#!/usr/bin/env python
""" Script to run the whole vault of regression tests.
"""
from __future__ import print_function
import pickle as pkl
import sys
import os

sys.path.insert(0, '../_modules')
from config import python2_exec
from config import python3_exec

# Reconstruct directory structure and edits to PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/testing/regression', '')

PYTHON_VERSION = sys.version_info[0]

################################################################################
# Compile
################################################################################
if PYTHON_VERSION == 2:
    python_exec = python2_exec
else:
    python_exec = python3_exec

# We need to be explicit about the PYTHON version as otherwise the F2PY
# libraries are not compiled in accordance with the PYTHON version used by
# for the execution of the script.
# cwd = os.getcwd()
# os.chdir(PACKAGE_DIR + '/respy')
# subprocess.check_call(python_exec + ' waf distclean', shell=True)
# subprocess.check_call(python_exec + ' waf configure build --debug',
#                 shell=True)
# os.chdir(cwd)

# Import package. The late import is required as the compilation needs to
# take place first.
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR

################################################################################
# RUN
################################################################################
fname = 'test_vault_' + str(PYTHON_VERSION) + '.respy.pkl'
tests_old = pkl.load(open(TEST_RESOURCES_DIR + '/' + fname, 'rb'))

tests_new = []
for idx, _ in enumerate(tests_old):
    print('\n Modfiying Test ', idx, 'with version ', PYTHON_VERSION)
    init_dict, crit_val = tests_old[idx]

    init_dict['SOLUTION']['store'] = False

    tests_new += [(init_dict, crit_val)]

pkl.dump(tests_new, open(fname, 'wb'))