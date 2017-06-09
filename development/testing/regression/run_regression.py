#!/usr/bin/env python
""" This script checks the regression tests vault for any unintended changes during further
development and refactoring efforts.
"""
from __future__ import print_function

from functools import partial

import multiprocessing as mp
import numpy as np
import argparse
import socket
import json

from auxiliary_shared import send_notification
from auxiliary_shared import compile_package

from auxiliary_regression import create_single
from auxiliary_regression import check_single
from auxiliary_regression import get_chunks

from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import print_init_dict
from codes.auxiliary import simulate_observed

HOSTNAME = socket.gethostname()


def run(request, is_compile, is_background, is_strict, num_procs):
    """ Run the regression tests.
    """
    if is_compile:
        compile_package(True)

    # We can set up a multiprocessing pool right away.
    mp_pool = mp.Pool(num_procs)

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

        # TODO: Create new set of regression tests.
        for name in ['OCCUPATION A', 'OCCUPATION B']:
            init_dict[name]['coeffs'].insert(8, 0.00)
            init_dict[name]['bounds'].insert(8, [None, None])
            init_dict[name]['fixed'].insert(8, True)

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
        # We maintain the separate execution in the case of a single processor for debugging
        # purposes. The error messages are generally much more informative.
        if num_procs == 1:
            tests = []
            for idx in range(num_tests):
                tests += [create_single(idx)]
        else:
            tests = mp_pool.map(create_single, range(num_tests))

        json.dump(tests, open('regression_vault.respy.json', 'w'))
        return

    if is_check:
        fname = TEST_RESOURCES_DIR + '/regression_vault.respy.json'
        tests = json.load(open(fname, 'r'))

        run_single = partial(check_single, tests)
        indices = list(range(num_tests))

        # We maintain the separate execution in the case of a single processor for debugging
        # purposes. The error messages are generally much more informative.
        if num_procs == 1:
            ret = []
            for index in indices:
                ret += [run_single(index)]
                # We need an early termination if a strict test run is requested.
                if is_strict and (False in ret):
                    break
        else:
            ret = []
            for chunk in get_chunks(indices, num_procs):
                ret += mp_pool.map(run_single, chunk)
                # We need an early termination if a strict test run is requested. So we check
                # whether there are any failures in the last batch.
                if is_strict and (False in ret):
                    break

        # This allows to call this test from another script, that runs other tests as well.
        is_failure, idx_failures = False, [i for i, x in enumerate(ret) if x is False]

        if len(idx_failures) > 0:
            is_failure = True

        if not is_background:
            send_notification('regression', is_failed=is_failure, idx_failures=idx_failures)

        return not is_failure


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create or check regression vault')

    parser.add_argument('--request', action='store', dest='request', help='task to perform',
                        required=True, nargs=2)

    parser.add_argument('--background', action='store_true', dest='is_background', default=False,
                        help='background process')

    parser.add_argument('--compile', action='store_true', dest='is_compile', default=False,
                        help='compile RESPY package')

    parser.add_argument('--strict', action='store_true', dest='is_strict', default=False,
                        help='immediate termination if failure')

    parser.add_argument('--procs', action='store', dest='num_procs', default=1, type=int,
                        help='number of processors')


    args = parser.parse_args()
    request, is_compile = args.request, args.is_compile,

    if is_compile:
        raise AssertionError('... probably not working at this point due to reload issues.')


    is_background = args.is_background
    is_strict = args.is_strict
    num_procs = args.num_procs

    run(request, is_compile, is_background, is_strict, num_procs)
