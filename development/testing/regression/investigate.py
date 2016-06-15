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

# Reconstruct directory structure and edits to PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/testing/regression', '')

PYTHON_VERSION = sys.version_info[0]

################################################################################
# Compile
################################################################################
python3_bin = "/home/peisenha/.envs/restudToolbox3/bin/python"
python2_bin = "/home/peisenha/.envs/restudToolbox2/bin/python"

if PYTHON_VERSION == 2:
    python_bin = python2_bin
else:
    python_bin = python3_bin

# We need to be explicit about the PYTHON version as otherwise the F2PY
# libraries are not compiled in accordance with the PYTHON version used by
# for the execution of the script.
if True:
    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR + '/respy')
    subprocess.call(python_bin + ' waf distclean', shell=True)
    subprocess.call(python_bin + ' waf configure build --debug --without_mpi',
        shell=True)
    os.chdir(cwd)

# Import package. The late import is required as the compilation needs to
# take place first.
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import print_init_dict

from respy import RespyCls
from respy import simulate
from respy import estimate

################################################################################
# RUN
################################################################################
idx = 499

fname = 'test_vault_' + str(PYTHON_VERSION) + '.respy.pkl'
tests = pkl.load(open(TEST_RESOURCES_DIR + '/' + fname, 'rb'))
print('\n Test ', idx, 'with version ', PYTHON_VERSION)
init_dict, crit_val = tests[idx]
print_init_dict(init_dict)
respy_obj = RespyCls('test.respy.ini')
simulate(respy_obj)
np.testing.assert_almost_equal(estimate(respy_obj)[1], crit_val)

