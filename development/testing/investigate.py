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
from modules.auxiliary import get_random_request
from modules.auxiliary import get_test_dict
from modules.auxiliary import cleanup

# Reconstruct directory structure and edits to PYTHONPATH
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROBUPY_DIR = BASE_DIR.replace('development/testing', '')
TEST_DIR = ROBUPY_DIR + '/robupy/tests'

sys.path.insert(0, ROBUPY_DIR)
sys.path.insert(0, TEST_DIR)


''' Request
'''
seed = 19387


# Cleanup
cleanup(False)


''' Error Reproduction
'''
np.random.seed(seed)

# Construct test
test_dict = get_test_dict(TEST_DIR)
module, method = get_random_request(test_dict)
mod = importlib.import_module(module)
test = getattr(mod.TestClass(), method)

test()
