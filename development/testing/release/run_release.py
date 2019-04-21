""" This script allows to test alternative releases against each other that are supposed
to lead to the same results for selected requests.
"""
import argparse
import os
import pickle as pkl
import random
import subprocess
from datetime import datetime
from datetime import timedelta

import numpy as np

from development.modules.auxiliary_release import prepare_release_tests
from development.modules.auxiliary_shared import cleanup
from development.modules.auxiliary_shared import send_notification
from respy import RespyCls

SCRIPT_FNAME = "../../modules/auxiliary_release.py"


def run(request, is_create, is_background, old_release, new_release):
    """ Test the different releases against each other.
    """
    cleanup()

    # Processing of command line arguments.
    if request[0] == "investigate":
        is_investigation, is_run = True, False
    elif request[0] == "run":
        is_investigation, is_run = False, True
    else:
        raise AssertionError("request in [run, investigate]")

    seed_investigation, hours = None, 0.0
    if is_investigation:
        seed_investigation = int(request[1])
        assert isinstance(seed_investigation, int)
    elif is_run:
        hours = float(request[1])
        assert hours > 0.0

    # Set up auxiliary information to construct commands.
    env_dir = os.environ["HOME"] + "/.envs"
    old_exec = env_dir + "/" + old_release + "/bin/python"
    new_exec = env_dir + "/" + new_release + "/bin/python"

    # Create fresh virtual environments if requested.
    if is_create:
        for release in [old_release, new_release]:
            cmd = ["virtualenv", env_dir + "/" + release, "--clear"]
            subprocess.check_call(cmd)

        # Set up the virtual environments with the two releases under
        # investigation.
        for which in ["old", "new"]:
            if which == "old":
                release, python_exec = old_release, old_exec
            elif which == "new":
                release, python_exec = new_release, new_exec
            else:
                raise AssertionError

            cmd = [python_exec, SCRIPT_FNAME, "upgrade"]
            subprocess.check_call(cmd)

            cmd = [python_exec, SCRIPT_FNAME, "prepare", release]
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

        # The idea is to have all elements that are hand-crafted for the release comparison in
        # the function below.
        constr = {}
        constr["flag_estimation"] = True

        prepare_release_tests(constr, old_release, new_release)

        # We use the current release for the simulation of the underlying dataset.
        respy_obj = RespyCls("test.respy.ini")
        respy_obj.simulate()

        for which in ["old", "new"]:

            if which == "old":
                release, python_exec = old_release, old_exec
            elif which == "new":
                release, python_exec = old_release, new_exec
            else:
                raise AssertionError

            cmd = [python_exec, SCRIPT_FNAME, "estimate", which]
            subprocess.check_call(cmd)

        # Compare the resulting values of the criterion function.
        crit_val_old = pkl.load(open("old/crit_val.respy.pkl", "rb"))
        crit_val_new = pkl.load(open("new/crit_val.respy.pkl", "rb"))

        if not is_investigation:
            try:
                np.testing.assert_allclose(crit_val_old, crit_val_new)
            except AssertionError:
                is_failure = True
        else:
            np.testing.assert_allclose(crit_val_old, crit_val_new)

        is_timeout = timeout < datetime.now() - start

        if is_investigation or is_failure or is_timeout:
            break

    if not is_background and not is_investigation:
        send_notification(
            "release",
            hours=hours,
            is_failed=is_failure,
            seed=seed,
            num_tests=num_tests,
            old_release=old_release,
            new_release=new_release,
        )


if __name__ == "__main__":

    # NOTES:
    #
    # All the versions are available on PYPI. Those marked as development versions are not
    # downloaded by an unsuspecting user. Instead, they need to be explicitly requested. Please
    # also make sure these versions are tagged in GIT.
    #
    #   Version     Description
    #
    #   1.0.0       Release for risk-only model
    #
    #   2.0.0.dev7  Ambiguity, probably same as v2.0.4
    #
    #   2.0.0.dev8  Ambiguity, now with free variances in worst-case determination
    #
    #   2.0.0.dev9  We added additional output for simulated datasets and included a script to
    #               gently terminate an estimation.
    #
    #   2.0.0.dev10 We added additional information to the simulated dataset such as all
    #               simulated rewards available for the individual each period. We also added the
    #               ability to scale each coefficient simply by their magnitudes.
    #
    #   2.0.0.dev11 We added sheepskin effects and separate reentry cost by level of education.
    #
    #   2.0.0.dev12 We added a first-experience effect and scaled the squared term for the
    #               experience variable by 100.
    #
    #   2.0.0.dev13 Flawed submission, simply just ignore.
    #
    #   2.0.0.dev14 We added initial conditions and unobserved type heterogeneity. It needs to be
    #               checked against 2.0.0.dev14
    #
    #   2.0.0.dev15 This is a nuisance release as we did some minor work in the develop branch at
    #               did not create a new branch right away.
    #
    #   2.0.0.dev16 We added age/period effects to each of the rewards.
    #
    #   2.0.0.dev17 We added state variables that capture last year's occupation.
    #
    #   2.0.0.dev18 We now allow for conditional type probabilities.
    #
    #   2.0.0.dev19 We now have more general rewards, i.e. not income maximizing.
    #
    #   2.0.0.dev19 We just worked on some issues to cleanup the code base. We did not find any
    #               errors.
    #
    # The two releases that are tested against each other. These are downloaded from PYPI in
    # their own virtual environments.
    old_release, new_release = "2.0.0.dev19", "2.0.0.dev20"

    parser = argparse.ArgumentParser(description="Run release testing.")

    parser.add_argument(
        "--request",
        action="store",
        dest="request",
        help="task to perform",
        nargs=2,
        required=True,
    )

    parser.add_argument(
        "--create",
        action="store_true",
        dest="is_create",
        default=False,
        help="create new virtual environments",
    )

    parser.add_argument(
        "--background",
        action="store_true",
        dest="is_background",
        default=False,
        help="background process",
    )

    # Distribute arguments
    args = parser.parse_args()
    request, is_create = args.request, args.is_create
    is_background = args.is_background

    run(request, is_create, is_background, old_release, new_release)
