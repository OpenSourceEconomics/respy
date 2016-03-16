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

# project
from modules.auxiliary import distribute_input
from modules.auxiliary import start_logging
from modules.auxiliary import cleanup
from modules.auxiliary import finish

#import modules.battery as development_tests


# virtual environment
if not hasattr(sys, 'real_prefix'):
    raise AssertionError('Please use a virtual environment for testing')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROBUPY_DIR = BASE_DIR.replace('development/testing', '')

TEST_DIR = ROBUPY_DIR + '/robupy/tests'

sys.path.insert(0, BASE_DIR)

# ROBUPY import
sys.path.insert(0, ROBUPY_DIR)

# ROBPUPY testing codes
sys.path.insert(0, TEST_DIR)
from codes.auxiliary import build_f2py_testing
from codes.auxiliary import compile_package

from modules.auxiliary import get_random_request
from modules.auxiliary import get_test_dict

''' Main Function.
'''


def run(hours):
    """ Run test battery.
    """

    # Get a dictionary with all candidate test cases.
    test_dict = get_test_dict(TEST_DIR, BASE_DIR)

    # Start with a clean slate.
    cleanup(False)

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

        np.random.seed(23)

        # Construct test case
        module, method = get_random_request(test_dict)
        mod = importlib.import_module(module)
        test = getattr(mod.TestClass(), method)

        # Run random tes
        try:
            test()
        except Exception:
            msg = traceback.format_exc()

        #  Timeout.
        if timeout < datetime.now() - start:
            break

    # Finishing.
    return None


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

    # Ensure that the FORTRAN routines are available.
    if False:
        compile_package('--fortran --debug', True)
        build_f2py_testing(True)

    hours, notification = distribute_input(parser)

    #start_logging()


    dict_ = run(hours)

    cleanup(True)

    #finish(dict_, hours, notification)
