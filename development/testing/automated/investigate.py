#!/usr/bin/env python
""" Script to quickly investigate failed estimation runs.
"""
# standard library
import numpy as np

import importlib
import sys
import os

# Testing infrastructure
from modules.auxiliary import cleanup_testing_infrastructure
from modules.auxiliary import get_random_request
from modules.auxiliary import get_test_dict

# Reconstruct directory structure and edits to PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/testing/automated', '')

# ROBPUPY testing codes. The import of the PYTEST configuration file ensures
# that the PYTHONPATH is modified to allow for the use of the tests..
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, PACKAGE_DIR + 'respy/tests')

VERSIONS = ['PYTHON', 'FORTRAN', 'F2PY']


''' Request
'''
seed = 71150 # 6216748723


''' Error Reproduction
'''
cleanup_testing_infrastructure(True)

np.random.seed(seed)

# Construct test
for _ in range(1):
    test_dict = get_test_dict(PACKAGE_DIR + '/respy/tests')
    module, method = get_random_request(test_dict)

    module, method = 'test_integration', 'test_8'
    print(module, method)
    mod = importlib.import_module(module)
    test = getattr(mod.TestClass(), method)

    test()

