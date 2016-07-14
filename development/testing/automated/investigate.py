#!/usr/bin/env python
""" Script to quickly investigate failed estimation runs.
"""
import numpy as np
import subprocess
import importlib
import sys
import os

sys.path.insert(0, '../_modules')
from auxiliary_automated import cleanup_testing_infrastructure
from auxiliary_automated import get_random_request
from auxiliary_automated import get_test_dict

# Reconstruct directory structure and edits to PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/testing/automated', '')

# ROBUPY testing codes. The import of the PYTEST configuration file ensures
# that the PYTHONPATH is modified to allow for the use of the tests..
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, PACKAGE_DIR + 'respy/tests')

# Recompiling during debugging
if False:
    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR + '/respy')
    subprocess.check_call('./waf distclean', shell=True)
    subprocess.check_call('./waf configure build --debug', shell=True)
    os.chdir(cwd)
else:
	print('not recompiling')

''' Request '''
#MODULE test_parallelism METHOD test_1 SEED: 24029

seed = 123 # 6216748723


''' Error Reproduction '''
cleanup_testing_infrastructure(True)

np.random.seed(seed)

# Construct test
test_dict = get_test_dict(PACKAGE_DIR + '/respy/tests')
module, method = get_random_request(test_dict)

module, method = 'test_parallelism', 'test_2'
count = 0
os.system('git clean -d -f')
for i in range(1000):
	print(module, method)
	mod = importlib.import_module(module)
	test = getattr(mod.TestClass(), method)

	test()
	count = count +1
	print('completed ', count)

	#os.system('git clean -d -f')
