#!/usr/bin/env python
""" Script to start development test battery for the ROBUPY package.
"""

# standard library
from datetime import timedelta
from datetime import datetime

import numpy as np

import traceback
import importlib
import argparse
import random
import sys
import os

# virtual environment
if not hasattr(sys, 'real_prefix'):
    raise AssertionError('Please use a virtual environment for testing')

# ROBPUPY testing codes. The import of the PYTEST configuration file ensures
# that the PYTHONPATH is modified to allow for the use of the tests..
TEST_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DIR = TEST_DIR.replace('development/testing', '') + '/robupy/tests'

sys.path.insert(0, TEST_DIR)
import conftest
from codes.auxiliary import cleanup_robupy_package
from codes.auxiliary import build_testing_library
from codes.auxiliary import build_robupy_package

# Testing infrastructure
from modules.auxiliary import cleanup_testing_infrastructure
from modules.auxiliary import get_random_request
from modules.auxiliary import distribute_input
from modules.auxiliary import get_test_dict

from modules.auxiliary import finalize_testing_record
from modules.auxiliary import update_testing_record
from modules.auxiliary import send_notification

''' Main Function.
'''


def run(hours):
    """ Run test battery.
    """

    # Get a dictionary with all candidate test cases.
    test_dict = get_test_dict(TEST_DIR)

    # We initialize a dictionary that allows to keep track of each test's
    # success or failure.
    full_test_record = dict()
    for key_ in test_dict.keys():
        full_test_record[key_] = dict()
        for value in test_dict[key_]:
            full_test_record[key_][value] = [0, 0]

    # Start with a clean slate.
    cleanup_testing_infrastructure(False)

    start, timeout = datetime.now(), timedelta(hours=hours)

    # # Evaluation loop.
    while True:

        # Set seed.
        seed = random.randrange(1, 100000)

        np.random.seed(seed)

        # Construct test case.
        module, method = get_random_request(test_dict)
        mod = importlib.import_module(module)
        test = getattr(mod.TestClass(), method)

        # Run random tes
        is_success, msg = None, None

        try:
            test()
            full_test_record[module][method][0] += 1
            is_success = True
        except Exception:
            full_test_record[module][method][1] += 1
            msg = traceback.format_exc()
            is_success = False

        # Record iteration
        update_testing_record(module, method, seed, is_success, msg,
                              full_test_record, start, timeout)

        cleanup_testing_infrastructure(True)

        #  Timeout.
        if timeout < datetime.now() - start:
            break

    finalize_testing_record()

''' Execution of module as script.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run development test '
                'battery of ROBUPY package.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--hours', action='store', dest='hours',
                        type=float, default=1.0, help='run time in hours')

    parser.add_argument('--notification', action='store_true',
                        dest='notification', default=False,
                        help='send notification')

    # Start from a clean slate and extract a user's request.
    cleanup_testing_infrastructure(False)
    hours, notification = distribute_input(parser)

    # Ensure that the FORTRAN resources are available. Some selected
    # functions are only used for testing purposes and thus collected in a
    # special FORTRAN library. The build of the ROBUPY package starts from a
    # clean slate and then the testing library is added to tests/lib directory.
    cleanup_robupy_package()
    build_robupy_package(True)
    build_testing_library(True)

    # Run testing infrastructure and send a notification (if requested).
    run(hours)

    if notification:
        send_notification(hours)

