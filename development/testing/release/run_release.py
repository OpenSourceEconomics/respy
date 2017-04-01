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
import os

from respy import RespyCls
from respy import simulate

from auxiliary_release import prepare_release_tests
from auxiliary_shared import send_notification
from auxiliary_shared import cleanup

SCRIPT_FNAME = '../../_modules/auxiliary_release.py'


def run(request, is_create, is_background, old_release, new_release):
    """ Test the different releases against each other.
    """
    cleanup()

    # Processing of command line arguments.
    if request[0] == 'investigate':
        is_investigation, is_run = True, False
    elif request[0] == 'run':
        is_investigation, is_run = False, True
    else:
        raise AssertionError('request in [run, investigate]')

    seed_investigation, hours = None, 0.0
    if is_investigation:
        seed_investigation = int(request[1])
        assert isinstance(seed_investigation, int)
    elif is_run:
        hours = float(request[1])
        assert (hours > 0.0)

    # Set up auxiliary information to construct commands.
    env_dir = os.environ['HOME'] + '/.envs'
    old_exec = env_dir + '/' + old_release + '/bin/python'
    new_exec = env_dir + '/' + old_release + '/bin/python'

    # Create fresh virtual environments if requested.
    if is_create:
        for release in [old_release, old_release]:
            cmd = ['virtualenv', env_dir + '/' + release, '--clear']
            subprocess.check_call(cmd)

        # Set up the virtual environments with the two releases under
        # investigation.
        for which in ['old', 'new']:
            if which == 'old':
                release, python_exec = old_release, old_exec
            elif which == 'new':
                release, python_exec = new_release, new_exec
            else:
                raise AssertionError

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
        constr['flag_estimation'] = True

        prepare_release_tests(constr, old_release, new_release)

        # We use the current release for the simulation of the underlying
        # dataset.
        respy_obj = RespyCls('test.respy.ini')
        simulate(respy_obj)

        for which in ['old', 'new']:

            if which == 'old':
                release, python_exec = old_release, old_exec
            elif which == 'new':
                release, python_exec = old_release, new_exec
            else:
                raise AssertionError

            cmd = [python_exec, SCRIPT_FNAME, 'estimate', which]
            subprocess.check_call(cmd)

        # Compare the resulting values of the criterion function.
        crit_val_old = pkl.load(open('old/crit_val.respy.pkl', 'rb'))
        crit_val_new = pkl.load(open('new/crit_val.respy.pkl', 'rb'))

        try:
            np.testing.assert_allclose(crit_val_old, crit_val_new)
        except AssertionError:
            is_failure = True

        is_timeout = timeout < datetime.now() - start

        if is_investigation or is_failure or is_timeout:
            break

    if not is_background and not is_investigation:
        send_notification('release', hours=hours, is_failed=is_failure,
            seed=seed, num_tests=num_tests, old_release=old_release,
            new_release=old_release)

if __name__ == '__main__':

    # NOTES:
    #
    #   All the versions are available on PYPI. Those marked as development
    #   versions are not downloaded by an unsuspecting user. Instead, they
    #   need to be explicitly requested. Please also make sure these versions
    #   are tagged in GIT.
    #
    #   Version     Description
    #
    #   1.0.0       Release for risk-only model
    #
    #   2.0.0.dev7  Ambiguity, probably same as v2.0.4
    #
    #   2.0.0.dev8  Ambiguity, now with free variances in worst-case
    #               determination
    #
    #   2.0.0.dev9  We added additional output for simulated datasets and
    #               included a script to gently terminate an estimation.
    #
    #   2.0.0.dev10 We added additional information to the simulated dataset
    #               such as all simulated rewards available for the individual
    #               each period. We also added the ability to scale each
    #               coefficient simply by their magnitudes.
    #
    #   TODO:   Isn't it a better way to have the candidate branch not set up
    #           as a release from PYPI as well?
    #
    # The two releases that are tested against each other. These are
    # downloaded from PYPI in their own virtual environments.
    old_release, new_release = '2.0.0.dev10', '2.0.0.dev11'

    parser = argparse.ArgumentParser(description='Run release testing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--request', action='store', dest='request',
        help='task to perform', nargs=2, required=True)

    parser.add_argument('--create', action='store_true', dest='is_create',
        default=False, help='create new virtual environments')

    parser.add_argument('--background', action='store_true',
        dest='is_background', default=False, help='background process')

    # Distribute arguments
    args = parser.parse_args()
    request, is_create = args.request, args.is_create
    is_background = args.is_background

    run(request, is_create, is_background, old_release, new_release)
