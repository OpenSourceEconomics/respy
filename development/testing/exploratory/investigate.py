#!/usr/bin/env python
""" Script to quickly investigate failed estimation runs.
"""
# standard library
import numpy as np

import importlib
import sys
import os

# Testing infrastructure
#from modules.auxiliary import cleanup_testing_infrastructure
#from modules.auxiliary import get_random_request
#from modules.auxiliary import get_test_dict

# Reconstruct directory structure and edits to PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/testing/exploratory', '')

# ROBPUPY testing codes. The import of the PYTEST configuration file ensures
# that the PYTHONPATH is modified to allow for the use of the tests..
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, PACKAGE_DIR + 'respy/tests')

# Check the full test vault.
from respy.python.shared.shared_auxiliary import print_init_dict

from respy.evaluate import evaluate
from respy.solve import solve

from respy import RespyCls
from respy import simulate



from respy.python.shared.shared_constants import TEST_RESOURCES_DIR

import pickle as pkl
version = str(sys.version_info[0])
fname = 'test_vault_' + version + '.respy.pkl'
tests = pkl.load(open(TEST_RESOURCES_DIR + '/' + fname, 'rb'))

print(fname)
new_tests = []

np.random.seed(123)
for idx in range(1000):
    print(idx)
    init_dict, crit_val = tests[idx]

    print_init_dict(init_dict)
    respy_obj = RespyCls('test.respy.ini')
    simulate(respy_obj)
    np.testing.assert_almost_equal(evaluate(respy_obj), crit_val)
