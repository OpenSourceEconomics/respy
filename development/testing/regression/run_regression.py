"""Create, run or investigate regression checks."""
import argparse
import os
import pickle
import shutil
import socket
from functools import partial
from multiprocessing import Pool

import numpy as np

import respy as rp
from development.modules.auxiliary_shared import get_random_dirname
from development.modules.auxiliary_shared import send_notification
from respy.config import DECIMALS
from respy.config import TEST_RESOURCES_DIR
from respy.config import TOL
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data

HOSTNAME = socket.gethostname()


def run_regression_tests(num_tests=None, tests=None, num_procs=1, strict=False):
    """Run regression tests.

    Args:
        num_tests (int): number of tests to run. If None, all are run.
        tests (list): list of regression tests. If None, tests are loaded from disk.
        num_procs (int): number of processes. Default 1.

    """
    tests = load_regression_tests() if tests is None else tests
    tests = tests[:num_tests] if num_tests is not None else tests

    if num_procs == 1:
        ret = []
        for test in tests:
            ret.append(check_single(test, strict=strict))
    else:
        mp_pool = Pool(num_procs)
        check = partial(check_single, strict=strict)
        ret = mp_pool.map(check, tests)

    idx_failures = [i for i, x in enumerate(ret) if x is False]
    is_failure = len(idx_failures) > 0

    print(f"Failures: {idx_failures}")

    send_notification("regression", is_failed=is_failure, idx_failures=idx_failures)


def create_regression_tests(num_tests, num_procs=1, write_out=False):
    """Create a regression vault.

    Args:
        num_test (int): How many tests are in the vault.
        num_procs (int): Number of processes. Default 1.
        write_out (bool): If True, regression tests are stored to disk, replacing
            any existing regression tests. Be careful with this. Default False.

    """
    if num_procs == 1:
        tests = []
        for idx in range(num_tests):
            tests += [create_single(idx)]
    else:
        with Pool(num_procs) as p:
            tests = p.map(create_single, range(num_tests))

    if write_out is True:
        with open(TEST_RESOURCES_DIR / "regression_vault.pickle", "wb") as p:
            pickle.dump(tests, p)
    return tests


def load_regression_tests():
    """Load regression tests from disk."""
    with open(TEST_RESOURCES_DIR / "regression_vault.pickle", "rb") as p:
        tests = pickle.load(p)
    return tests


def investigate_regression_test(idx):
    """Investigate regression tests."""
    tests = load_regression_tests()
    attr, crit_val = tests[idx]
    df = simulate_truncated_data(attr)

    x = rp.get_parameter_vector(attr)
    crit_func = rp.get_crit_func(attr, df)

    result = crit_func(x)

    np.testing.assert_almost_equal(result, crit_val, decimal=DECIMALS)


def check_single(test, strict=False):
    """Check a single test."""
    param_spec, option_spec, crit_val = test

    # We need to create an temporary directory, so the multiprocessing does not
    # interfere with any of the files that are printed and used during the small
    # estimation request.
    dirname = get_random_dirname(5)
    os.mkdir(dirname)
    os.chdir(dirname)

    df = simulate_truncated_data(param_spec, option_spec)

    crit_func = rp.get_crit_func(param_spec, option_spec, df)

    est_val = crit_func(param_spec)
    is_success = np.isclose(est_val, crit_val, rtol=TOL, atol=TOL)

    if strict is True:
        assert is_success, "Failed regression test."

    # Cleanup of temporary directories.from
    os.chdir("..")
    shutil.rmtree(dirname)

    assert is_success


def create_single(idx):
    """Create a single test."""
    dirname = get_random_dirname(5)
    os.mkdir(dirname)
    os.chdir(dirname)
    np.random.seed(idx)
    param_spec, options_spec = generate_random_model()
    df = simulate_truncated_data(param_spec, options_spec)

    crit_func = rp.get_crit_func(param_spec, options_spec, df)

    crit_val = crit_func(param_spec)

    if not isinstance(crit_val, float):
        raise AssertionError(" ... value of criterion function too large.")
    os.chdir("..")
    shutil.rmtree(dirname)
    return param_spec, options_spec, crit_val


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
    request = args.request
    is_strict = args.is_strict
    num_procs = args.num_procs

    if request[0] == "run":
        run_regression_tests(
            num_tests=int(request[1]), num_procs=num_procs, strict=is_strict
        )
    elif request[0] == "investigate":
        investigate_regression_test(int(request[1]))
    elif request[0] == "create":
        create_regression_tests(
            num_tests=int(request[1]), num_procs=num_procs, write_out=True
        )
    else:
        raise ValueError("Invalid request.")
