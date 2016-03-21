#!/usr/bin/env python
""" Script to quickly investigate failed estimation runs.
"""
# standard library
import numpy as np

import importlib
import sys
import os

# virtual environment
if not hasattr(sys, 'real_prefix'):
    raise AssertionError('Please use a virtual environment for testing')


# Testing infrastructure
from modules.auxiliary import cleanup_testing_infrastructure
from modules.auxiliary import get_random_request
from modules.auxiliary import get_test_dict

# Reconstruct directory structure and edits to PYTHONPATH
TEST_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DIR = TEST_DIR.replace('development/testing', '')
TEST_DIR = TEST_DIR + '/robupy/tests'

# ROBPUPY testing codes. The import of the PYTEST configuration file ensures
# that the PYTHONPATH is modified to allow for the use of the tests..
sys.path.insert(0, TEST_DIR)
import conftest
from codes.auxiliary import cleanup_robupy_package
from codes.auxiliary import build_testing_library
from codes.auxiliary import build_robupy_package

''' Request
'''
seed = 1793

if False:
    build_robupy_package(False)
#    build_testing_library(False)

''' Error Reproduction
'''
cleanup_testing_infrastructure(True)

np.random.seed(seed)

# Construct test
test_dict = get_test_dict(TEST_DIR)
module, method = get_random_request(test_dict)
print(module, method)
mod = importlib.import_module(module)
test = getattr(mod.TestClass(), method)

test()
