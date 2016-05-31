#!/usr/bin/env python
""" Script to run the whole vault of regression tests.
"""
# standard library
import pickle as pkl
import numpy as np
import sys
import os

# Reconstruct directory structure and edits to PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/testing/exploratory', '')

# Check the full test vault.
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.evaluate import evaluate
from respy import RespyCls
from respy import simulate

################################################################################
# RUN
################################################################################
fname = 'test_vault_' + str(sys.version_info[0]) + '.respy.pkl'
tests = pkl.load(open(TEST_RESOURCES_DIR + '/' + fname, 'rb'))

for idx in range(len(tests)):
    print('\n Evaluation ', idx)
    init_dict, crit_val = tests[idx]
    print_init_dict(init_dict)
    respy_obj = RespyCls('test.respy.ini')
    simulate(respy_obj)
    np.testing.assert_almost_equal(evaluate(respy_obj), crit_val)
