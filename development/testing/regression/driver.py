#!/usr/bin/env python
import argparse
import numpy as np
import json

from auxiliary_shared import send_notification
from auxiliary_shared import compile_package

from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import print_init_dict

from codes.random_init import generate_init


def run(args):
    """ Run the regression tests.
    """
    if args.is_compile:
        compile_package(True)

    # The late import is required so a potentially just compiled FORTRAN
    # implementation is recognized. This is important for the creation of the
    # regression vault as we want to include FORTRAN use cases.
    from respy import RespyCls
    from respy import simulate
    from respy import estimate

    # Process command line arguments
    is_creation, is_modification = False, False
    is_investigation, is_check = False, False
    num_tests, idx = None, None

    if args.request[0] == 'create':
        is_creation, num_tests = True, int(args.request[1])
    elif args.request[0] == 'check':
        is_check, num_tests = True, int(args.request[1])
    elif args.request[0] == 'modify':
        is_modification, num_tests = True, int(args.request[1])
    elif args.request[0] == 'investigate':
        is_investigation, idx = True, int(args.request[1])
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

        simulate(respy_obj)
        np.testing.assert_almost_equal(estimate(respy_obj)[1], crit_val)

    if is_modification:
        fname = TEST_RESOURCES_DIR + '/regression_vault.respy.json'
        tests_old = json.load(open(fname, 'r'))
        # TODO: Is this True, cant we do that in the regression test file?
        # [str(i) for i in paras_fixed], see random_init
        tests_new = []
        for idx, _ in enumerate(tests_old):
            print('\n Modfiying Test ', idx)

            init_dict, crit_val = tests_old[idx]

            print(init_dict['PROGRAM']['version'])
            if init_dict['PROGRAM']['version'] == 'FORTRAN':
                continue
            # This is where the modifications take place
            tests_new += [(init_dict, crit_val)]

        print(len(tests_new))
        json.dump(tests_new, open('regression_vault.respy.json', 'w'))
        return

    if is_creation:
        # CHECK SOME TODOS.
        tests = []
        for idx in range(num_tests):
            print('\n Creating Test ', idx)

            # We impose a couple of constraints that make the requests
            # manageable.
            constr = dict()
            constr['maxfun'] = int(np.random.choice([0, 1, 2, 3, 5, 6], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
            constr['flag_scaling'] = np.random.choice([True, False], p=[0.1, 0.9])
            constr['is_store'] = False

            init_dict = generate_init(constr)
            respy_obj = RespyCls('test.respy.ini')
            simulate(respy_obj)
            crit_val = estimate(respy_obj)[1]
            test = (init_dict, crit_val)
            tests += [test]

        json.dump(tests, open('regression_vault.respy.json', 'w'))
        return

    if is_check:
        fname = TEST_RESOURCES_DIR + '/regression_vault.respy.json'
        tests = json.load(open(fname, 'r'))

        for idx in range(num_tests):
            print('\n Checking Test ', idx)

            init_dict, crit_val = tests[idx]

            print_init_dict(init_dict)
            respy_obj = RespyCls('test.respy.ini')

            simulate(respy_obj)
            np.testing.assert_almost_equal(estimate(respy_obj)[1], crit_val)

        # This allows to call this test from another script, that runs other
        # tests as well.
        if not args.is_background:
            send_notification('regression')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create or check both vaults',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--request', action='store', dest='request',
        help='task to perform', required=True, nargs=2)

    parser.add_argument('--background', action='store_true',
        dest='is_background', default=False, help='background process')

    parser.add_argument('--compile', action='store_true', dest='is_compile',
        default=False, help='compile RESPY package')

    run(parser.parse_args())
