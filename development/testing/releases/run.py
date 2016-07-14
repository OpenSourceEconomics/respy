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
import os

sys.path.insert(0, '../../../respy/tests')
from codes.random_init import generate_init
from respy import RespyCls
from respy import simulate

sys.path.insert(0, '../_modules')
from auxiliary_shared import send_notification
from auxiliary_shared import cleanup


def run(args):

    # Set up auxiliary information to construct commands.
    env_dir = os.environ['HOME'] + '/.envs'
    old_exec = env_dir + '/' + OLD_RELEASE + '/bin/python'
    new_exec = env_dir + '/' + NEW_RELEASE + '/bin/python'

    # Create fresh virtual environments.
    for release in [OLD_RELEASE, NEW_RELEASE]:
        cmd = ['pyvenv', env_dir + '/' + release, '--clear']
        subprocess.check_call(cmd)

    # Set up the virtual environments with the two releases under investigation.
    for which in ['old', 'new']:
        if which == 'old':
            release, python_exec = OLD_RELEASE, old_exec
        elif which == 'new':
            release, python_exec = NEW_RELEASE, new_exec
        else:
            raise AssertionError
        cmd = [python_exec, '../_modules/auxiliary_release.py', release]
        subprocess.check_call(cmd)

    # Run tests
    # TODO: How about log files.
    cleanup()
    is_failed = False
    # Evaluation loop.
    start, timeout = datetime.now(), timedelta(hours=args.hours)
    num_tests = 0

    is_running = True
    while is_running:
        num_tests += 1

        # Set seed.
        seed = random.randrange(1, 100000)
        np.random.seed(seed)

        # Create a random estimation task.
        constr = dict()
        constr['is_estimation'] = True

        generate_init(constr)

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
            cmd = [python_exec, '../_modules/auxiliary_release.py']
            subprocess.check_call(cmd)

            if crit_val is None:
                crit_val = np.genfromtxt('.crit_val')

            try:
                np.testing.assert_equal(crit_val, np.genfromtxt(
                    '.crit_val'))
            except AssertionError:
                is_failed = True
                is_running = False

        # Timeout.
        if timeout < datetime.now() - start:
            break

    send_notification('release', args.hours, is_failed, seed, num_tests)


if __name__ == '__main__':

    # The two releases that are tested against each other. These are
    # downloaded from PYPI in their own virtual environments.
    OLD_RELEASE, NEW_RELEASE = '0.0.19', '0.0.20'

    parser = argparse.ArgumentParser(description='Run testing of release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--hours', action='store', dest='hours',
        type=float, default=1.0, help='run time in hours')

    run(parser.parse_args())
