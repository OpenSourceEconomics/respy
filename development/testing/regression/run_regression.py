""" This script checks the regression tests vault for any unintended changes during
further development and refactoring efforts.
"""
from __future__ import print_function

from functools import partial

import multiprocessing as mp
import numpy as np
import argparse
import socket
import pickle

from development.modules.auxiliary_shared import send_notification
from development.modules.auxiliary_shared import compile_package

from development.modules.auxiliary_regression import create_single
from development.modules.auxiliary_regression import check_single
from development.modules.auxiliary_regression import get_chunks

from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_constants import DECIMALS
from respy.pre_processing.model_processing import _options_spec_from_attributes, _params_spec_from_attributes
from respy.tests.codes.auxiliary import simulate_observed

HOSTNAME = socket.gethostname()

from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.python.shared.shared_constants import IS_FORTRAN


def run(request, is_compile, is_background, is_strict, num_procs):
    """ Run the regression tests.
    """
    if is_compile:
        compile_package(True)

    # We can set up a multiprocessing pool right away.
    mp_pool = mp.Pool(num_procs)

    # The late import is required so a potentially just compiled FORTRAN implementation
    # is recognized. This is important for the creation of the regression vault as we
    # want to include FORTRAN use cases.
    from respy import RespyCls

    # Process command line arguments
    is_creation = False
    is_investigation, is_check = False, False
    num_tests, idx = None, None

    if request[0] == "create":
        is_creation, num_tests = True, int(request[1])
    elif request[0] == "check":
        is_check, num_tests = True, int(request[1])
    elif request[0] == "investigate":
        is_investigation, idx = True, int(request[1])
    else:
        raise AssertionError("request in [create, check. investigate]")
    if num_tests is not None:
        assert num_tests > 0
    if idx is not None:
        assert idx > 0

    if is_investigation:
        fname = TEST_RESOURCES_DIR / "regression_vault.pickle"
        with open(fname, 'rb') as p:
            tests = pickle.load(p)

        attr, crit_val = tests[idx]
        params_spec = _params_spec_from_attributes(attr)
        options_spec = _options_spec_from_attributes(attr)

        respy_obj = RespyCls(params_spec, options_spec)

        simulate_observed(respy_obj)

        result = respy_obj.fit()[1]
        np.testing.assert_almost_equal(result, crit_val, decimal=DECIMALS)

    if is_creation:
        # We maintain the separate execution in the case of a single processor for
        # debugging purposes. The error messages are generally much more informative.
        if num_procs == 1:
            tests = []
            for idx in range(num_tests):
                tests += [create_single(idx)]
        else:
            tests = mp_pool.map(create_single, range(num_tests))

        with open(TEST_RESOURCES_DIR / "regression_vault.pickle", "wb") as p:
            pickle.dump(tests, p)
        return

    if is_check:
        fname = TEST_RESOURCES_DIR / "regression_vault.pickle"
        with open(fname, 'rb') as p:
            tests = pickle.load(p)

        run_single = partial(check_single, tests)
        indices = list(range(num_tests))

        # We maintain the separate execution in the case of a single processor for
        # debugging purposes. The error messages are generally much more informative.
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
                # We need an early termination if a strict test run is requested. So we
                # check whether there are any failures in the last batch.
                if is_strict and (False in ret):
                    break

        # This allows to call this test from another script, that runs other tests as
        # well.
        idx_failures = [i for i, x in enumerate(ret) if x not in [True, None]]
        is_failure = False in ret

        if len(idx_failures) > 0:
            is_failure = True

        if not is_background:
            send_notification(
                "regression", is_failed=is_failure, idx_failures=idx_failures
            )

        return not is_failure


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create or check regression vault")

    parser.add_argument(
        "--request",
        action="store",
        dest="request",
        help="task to perform",
        required=True,
        nargs=2,
    )

    parser.add_argument(
        "--background",
        action="store_true",
        dest="is_background",
        default=False,
        help="background process",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        dest="is_compile",
        default=False,
        help="compile RESPY package",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        dest="is_strict",
        default=False,
        help="immediate termination if failure",
    )

    parser.add_argument(
        "--procs",
        action="store",
        dest="num_procs",
        default=1,
        type=int,
        help="number of processors",
    )

    args = parser.parse_args()
    request, is_compile = args.request, args.is_compile

    if is_compile:
        raise AssertionError(
            "... probably not working at this point due to reload issues."
        )

    is_background = args.is_background
    is_strict = args.is_strict
    num_procs = args.num_procs

    run(request, is_compile, is_background, is_strict, num_procs)
