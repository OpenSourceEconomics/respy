#!/usr/bin/env python
""" Script to start development test battery for the ROBUPY package.

    TODO: The testing routine does not account for problems in the
    optimization of the ambiguity step.

"""

# standard library
from datetime import timedelta
from datetime import datetime

import numpy as np

import traceback
import importlib
import argparse
import random
import glob
import sys
import os

#import modules.battery as development_tests


# virtual environment
if not hasattr(sys, 'real_prefix'):
    raise AssertionError('Please use a virtual environment for testing')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROBUPY_DIR = BASE_DIR.replace('development/testing', '')

TEST_DIR = ROBUPY_DIR + '/robupy/tests'

# ROBUPY import
#sys.path.insert(0, ROBUPY_DIR)

# ROBPUPY testing codes
sys.path.insert(0, TEST_DIR)
from codes.auxiliary import build_testing_library
from codes.auxiliary import build_robupy_package

# Testing infrastructure
from modules.auxiliary import cleanup_testing_infrastructure
from modules.auxiliary import get_random_request
from modules.auxiliary import distribute_input
from modules.auxiliary import get_test_dict
from modules.auxiliary import finish


''' Main Function.
'''


def run(hours):
    """ Run test battery.
    """

    # Get a dictionary with all candidate test cases.
    test_dict = get_test_dict(TEST_DIR)

    # We initialize a dictionary that allows to keep track of each test's
    # success or failure.
    test_record = dict()
    for key_ in test_dict.keys():
        test_record[key_] = dict()
        for value in test_dict[key_]:
            test_record[key_][value] = [0, 0]

    # Start with a clean slate.
    cleanup_testing_infrastructure(False)

    #
    start, timeout = datetime.now(), timedelta(hours=hours)

    # Initialize counter.
    # dict_ = dict()
    #
    # for label in labels:
    #
    #     dict_[label] = dict()
    #
    #     dict_[label]['success'] = 0
    #
    #     dict_[label]['failure'] = 0
    #
    # # Logging.
    # logger = logging.getLogger('DEV-TEST')
    #
    # msg = 'Initialization of a ' + str(hours) + ' hours testing run.'
    #
    # logger.info(msg)
    #
    # # Evaluation loop.
    while True:

        # Set seed.
        seed = random.randrange(1, 100000)

        np.random.seed(seed)

        # Construct test case
        module, method = get_random_request(test_dict)
        mod = importlib.import_module(module)
        test = getattr(mod.TestClass(), method)
        print(seed)
        # Run random tes
        try:
            test()

            test_record[module][method][0] += 1

        except Exception:
            test_record[module][method][1] += 1
        #    msg = traceback.format_exc()

        # Cleanup
        cleanup_testing_infrastructure(True)

        #  Timeout.
        if timeout < datetime.now() - start:
            break

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

    # Ensure that the FORTRAN resources are available. Some selected
    # functions are only used for testing purposes and thus collected in a
    # special FORTRAN library.
    build_testing_library(True)
    build_robupy_package(True)

    hours, notification = distribute_input(parser)

    #start_logging()


    dict_ = run(hours)

    cleanup_testing_infrastructure(True)

    finish(dict_, hours, notification)
