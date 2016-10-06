#!/usr/bin/env python
""" This script allows to test alternative releases against each other that
are supposed to lead to the same results for selected requests.
"""
from datetime import timedelta
from datetime import datetime

import pickle as pkl
import numpy as np

import subprocess
import argparse
import random
import sys
import os

# We are using features for the automatic creation of the virtual environment
# which are only available in Python 3.
if int(sys.version[0]) < 3:
    raise AssertionError('Please use Python 3')

from respy import RespyCls
from respy import simulate

from auxiliary_release import prepare_release_tests
from auxiliary_shared import send_notification
from auxiliary_shared import cleanup


SCRIPT_FNAME = '../../_modules/auxiliary_release.py'


def run(args):
    """ Test the different releases against each other.
    """
    cleanup()

    # Processing of command line arguments.
    if args.request[0] == 'investigate':
        is_investigation, is_run = True, False
    elif args.request[0] == 'run':
        is_investigation, is_run = False, True
    else:
        raise AssertionError('request in [run, investigate]')

    seed_investigation, hours = None, 0.0
    if is_investigation:
        seed_investigation = int(args.request[1])
        assert isinstance(seed_investigation, int)
    elif is_run:
        hours = float(args.request[1])
        assert (hours > 0.0)

    is_create = args.is_create

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
            # 8.1.1, which is the current default when invoking pyvenv. This
            # should simply solve itself over time.
            cmd = [python_exec, SCRIPT_FNAME, 'upgrade']
            subprocess.check_call(cmd)

            cmd = [python_exec, SCRIPT_FNAME, 'prepare', release]
            subprocess.check_call(cmd)

    # Evaluation loop.
    start, timeout = datetime.now(), timedelta(hours=hours)
    num_tests, is_failure = 0, False

    while True:
        num_tests += 1

        # Set seed.
        if is_investigation:
            seed = seed_investigation
        else:
            seed = random.randrange(1, 100000)

        np.random.seed(seed)

        # The idea is to have all elements that are hand-crafted for the
        # release comparison in the function below.
        constr = dict()
        constr['is_estimation'] = True

        prepare_release_tests(constr)

        # We use the current release for the simulation of the underlying
        # dataset.
        respy_obj = RespyCls('test.respy.ini')
        simulate(respy_obj)

        for which in ['old', 'new']:

            if which == 'old':
                release, python_exec = OLD_RELEASE, old_exec
            elif which == 'new':
                release, python_exec = NEW_RELEASE, new_exec
            else:
                raise AssertionError

            cmd = [python_exec, SCRIPT_FNAME, 'estimate', which]
            subprocess.check_call(cmd)

        # Compare the resulting values of the criterion function.
        crit_val_old = pkl.load(open('old/crit_val.respy.pkl', 'rb'))
        crit_val_new = pkl.load(open('new/crit_val.respy.pkl', 'rb'))

        #try:
        print('relevant seed', seed)
        np.testing.assert_allclose(crit_val_old, crit_val_new)
#        print('past testing', seed)
        #except AssertionError:
#            is_failure = True

        is_timeout = timeout < datetime.now() - start

        if is_investigation or is_failure or is_timeout:
            break

    if not is_investigation:
        send_notification('release', hours=hours, is_failed=is_failure,
             seed=seed, num_tests=num_tests)

if __name__ == '__main__':

    # The two releases that are tested against each other. These are
    # downloaded from PYPI in their own virtual environments.
    OLD_RELEASE, NEW_RELEASE = '1.0.00', '2.0.01'

    parser = argparse.ArgumentParser(description='Run release testing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--request', action='store', dest='request',
        help='task to perform', nargs=2, required=True)

    parser.add_argument('--create', action='store_true', dest='is_create',
        default=False, help='create new virtual environments')

    run(parser.parse_args())
