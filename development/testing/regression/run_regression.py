#!/usr/bin/env python
""" This script checks the regression tests vault for any unintended changes during further 
development and refactoring efforts.
"""
from __future__ import print_function

import numpy as np
import argparse
import socket
import json

from auxiliary_shared import send_notification
from auxiliary_shared import compile_package

from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_constants import IS_PARALLEL
from respy.python.shared.shared_constants import IS_FORTRAN
from codes.auxiliary import simulate_observed
from codes.random_init import generate_init

HOSTNAME = socket.gethostname()


def run(request, is_compile, is_background, is_strict):
    """ Run the regression tests.
    """
    if is_compile:
        compile_package(True)

    # The late import is required so a potentially just compiled FORTRAN implementation is
    # recognized. This is important for the creation of the regression vault as we want to
    # include FORTRAN use cases.
    from respy import RespyCls
    from respy import estimate

    # Process command line arguments
    is_creation, is_modification = False, False
    is_investigation, is_check = False, False
    num_tests, idx = None, None

    if request[0] == 'create':
        is_creation, num_tests = True, int(request[1])
    elif request[0] == 'check':
        is_check, num_tests = True, int(request[1])
    elif request[0] == 'modify':
        is_modification, num_tests = True, int(request[1])
    elif request[0] == 'investigate':
        is_investigation, idx = True, int(request[1])
    else:
        raise AssertionError('request in [create, check, modify. investigate]')
    if num_tests is not None:
        assert num_tests > 0
    if idx is not None:
        assert idx > 0

    if is_investigation:
        fname = TEST_RESOURCES_DIR + '/regression_vault.respy.json'
        tests = json.load(open(fname, 'r'))

        init_dict, crit_val = tests[idx]
        print_init_dict(init_dict)
        respy_obj = RespyCls('test.respy.ini')

        simulate_observed(respy_obj)
        np.testing.assert_almost_equal(estimate(respy_obj)[1], crit_val)

    if is_modification:
        fname = TEST_RESOURCES_DIR + '/regression_vault.respy.json'
        tests_old = json.load(open(fname, 'r'))

        tests_new = []
        for idx, _ in enumerate(tests_old):
            print('\n Modfiying Test ', idx)

            init_dict, crit_val = tests_old[idx]

            # This is where the modifications take place
            tests_new += [(init_dict, crit_val)]

        json.dump(tests_new, open('regression_vault.respy.json', 'w'))
        return

    if is_creation:
        tests = []
        for idx in range(num_tests):
            print('\n Creating Test ', idx)

            # We impose a couple of constraints that make the requests manageable.
            np.random.seed(idx)
            constr = dict()
            constr['flag_estimation'] = True

            init_dict = generate_init(constr)
            respy_obj = RespyCls('test.respy.ini')
            simulate_observed(respy_obj)
            crit_val = estimate(respy_obj)[1]
            test = (init_dict, crit_val)
            tests += [test]
            print_init_dict(init_dict)

        json.dump(tests, open('regression_vault.respy.json', 'w'))
        return

    if is_check:
        fname = TEST_RESOURCES_DIR + '/regression_vault.respy.json'
        tests = json.load(open(fname, 'r'))

        # We shuffle the order of the tests so checking subset is insightful.
        indices = list(range(num_tests))
        np.random.shuffle(indices)

        # We collect the indices for the failed tests which allows for easy investigation.
        idx_failures = []

        for i, idx in enumerate(indices):
            print('\n\n Checking Test ', idx, ' at iteration ',  i, '\n')

            init_dict, crit_val = tests[idx]

            # During development it is useful that we can only run the PYTHON versions of the
            # program.
            msg = ' ... skipped as required version of package not available'
            if init_dict['PROGRAM']['version'] == 'FORTRAN' and not IS_FORTRAN:
                print(msg)
                continue
            if init_dict['PROGRAM']['procs'] > 1 and not IS_PARALLEL:
                print(msg)
                continue

            # TODO: All other bounds are not enforcable at this point.
            num_types = len(init_dict['TYPE_SHARES']['coeffs'])
            init_dict['TYPE_SHARES']['bounds'] = [[0.0, None]] * num_types

            # This is the baseline code again.
            print_init_dict(init_dict)
            respy_obj = RespyCls('test.respy.ini')
            simulate_observed(respy_obj)

            est_val = estimate(respy_obj)[1]
            try:
                np.testing.assert_almost_equal(est_val, crit_val)
                print(' ... success')
            except AssertionError:
                print(' ..failure')
                idx_failures += [idx]
                # We do not always want to break immediately as for some reason a very small
                # number of tests might fail on a machine that was not used to create the test
                # vault.
                if is_strict:
                    break

        # This allows to call this test from another script, that runs other tests as well.
        is_failure = False
        if len(idx_failures) > 0:
            is_failure = True

        if not is_background:
            send_notification('regression', is_failed=is_failure, idx_failures=idx_failures)

        return not is_failure


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create or check both vaults')

    parser.add_argument('--request', action='store', dest='request', help='task to perform',
                        required=True, nargs=2)

    parser.add_argument('--background', action='store_true', dest='is_background', default=False,
                        help='background process')

    parser.add_argument('--compile', action='store_true', dest='is_compile', default=False,
                        help='compile RESPY package')

    parser.add_argument('--strict', action='store_true', dest='is_strict', default=False,
                        help='immediate termination if failure')

    args = parser.parse_args()
    request, is_compile = args.request, args.is_compile,
    is_background = args.is_background
    is_strict = args.is_strict

    run(request, is_compile, is_background, is_strict)
