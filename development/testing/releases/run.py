#!/usr/bin/env python
""" This script allows to test alternative releases against each other that
are supposed to lead to the same results for selected requests.
"""
from datetime import timedelta
from datetime import datetime
import numpy as np
import subprocess
import argparse
import random
import sys
import pickle as pkl
import os

sys.path.insert(0, '../../../respy/tests')
from codes.random_init import generate_init
from respy import RespyCls
from respy import simulate

sys.path.insert(0, '../_modules')
from auxiliary_shared import send_notification
from auxiliary_shared import cleanup

SCRIPT = '../_modules/auxiliary_release.py'


def run(args):
    """ Test the different releases against each other.
    """
    # Distribute arguments
    is_create = args.is_create
    hours = args.hours

    # Set up auxiliary information to construct commands.
    env_dir = os.environ['HOME'] + '/.envs'
    old_exec = env_dir + '/' + OLD_RELEASE + '/bin/python'
    new_exec = env_dir + '/' + NEW_RELEASE + '/bin/python'

    # Create fresh virtual environments if requested.
    if is_create:
        for release in [OLD_RELEASE, NEW_RELEASE]:
            cmd = ['pyvenv', env_dir + '/' + release, '--clear']
            subprocess.check_call(cmd)

        # Set up the virtual environments with the two releases under
        # investigation.
        for which in ['old', 'new']:
            if which == 'old':
                release, python_exec = OLD_RELEASE, old_exec
            elif which == 'new':
                release, python_exec = NEW_RELEASE, new_exec
            else:
                raise AssertionError

            # TODO: This is a temporary bug fix. The problem seems to be the
            # slightly outdated pip. It needs to be 8.1.2 instead of the
            # precursing reason. This should simply solve itself over time.
            cmd = [python_exec, SCRIPT, 'upgrade']
            subprocess.check_call(cmd)

            cmd = [python_exec, SCRIPT, 'prepare', release]
            subprocess.check_call(cmd)

    # # Run tests
    # # TODO: How about log files.
    cleanup()
    is_failed = False
    # Evaluation loop.
    start, timeout = datetime.now(), timedelta(hours=hours)
    num_tests = 0

    is_running = True
    while is_running:
        num_tests += 1

        print('\n\n')
        print(num_tests)
        print('\n\n')
        # Set seed.
        seed = random.randrange(1, 100000)
        np.random.seed(seed)

        # Create a random estimation task.
        constr = dict()
        constr['is_estimation'] = True

        # TODO: This is where some code that is specific to the release
        # comparison is put.
        constr['level'] = 0.00
        constr['flag_ambiguity'] = True

        init_dict = generate_init(constr)

        # TODO: Some modifications are required so that the old release is
        # working on the same model.
        init_dict['ESTIMATION']['tau'] = int(init_dict['ESTIMATION']['tau'])
        pkl.dump(init_dict, open('release_new.respy.pkl', 'wb'))

        del init_dict['AMBIGUITY']
        pkl.dump(init_dict, open('release_old.respy.pkl', 'wb'))

        # We use the current release for the simulation of the underlying
        # dataset.
        respy_obj = RespyCls('test.respy.ini')
        simulate(respy_obj)

        crit_val = None
        for which in ['old', 'new']:
            if which == 'old':
                release, python_exec = OLD_RELEASE, old_exec
            elif which == 'new':
                release, python_exec = NEW_RELEASE, new_exec
            else:
                raise AssertionError
            cmd = [python_exec, SCRIPT, 'estimate', which]
            subprocess.check_call(cmd)

            if crit_val is None:
                crit_val = np.genfromtxt('.crit_val')


            np.testing.assert_equal(crit_val, np.genfromtxt('.crit_val'))

                 #
    #         try:
    #             np.testing.assert_equal(crit_val, np.genfromtxt(
    #                 '.crit_val'))
    #         except AssertionError:
    #             is_failed = True
    #             is_running = False
    #
        # Timeout.
        if timeout < datetime.now() - start:
            break
    #
    # send_notification('release', hours=args.hours, is_failed=is_failed,
    #         seed=seed, num_tests=num_tests)
    #

if __name__ == '__main__':

    # The two releases that are tested against each other. These are
    # downloaded from PYPI in their own virtual environments.
    OLD_RELEASE, NEW_RELEASE = '1.0.00', '2.0.00'

    parser = argparse.ArgumentParser(description='Run testing of release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--hours', action='store', dest='hours',
        type=float, default=1.0, help='run time in hours')

    parser.add_argument('--create', action='store_true', dest='is_create',
        default=False, help='create new virtual environments')

    run(parser.parse_args())
