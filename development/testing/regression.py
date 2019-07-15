"""Create, run or investigate regression checks."""
import os
import pickle
import socket
from functools import partial
from multiprocessing import Pool

import click
import numpy as np

import respy as rp
from development.testing.notifications import send_notification
from respy.config import DECIMALS
from respy.config import TEST_RESOURCES_DIR
from respy.config import TOL
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data

HOSTNAME = socket.gethostname()


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def run_regression_tests(num_tests=None, num_procs=1, strict=False):
    """Run regression tests.

    Parameters
    ----------
    num_tests : int
        Number of tests to run. If None, all are run.
    tests : list
        List of regression tests. If None, tests are loaded from disk.
    num_procs : int
        Number of processes. Default 1.

    """
    tests = load_regression_tests()
    tests = tests[:num_tests] if num_tests is not None else tests

    if num_procs == 1:
        ret = []
        for test in tests:
            ret.append(check_single(test, strict=strict))
    else:
        mp_pool = Pool(num_procs)
        check = partial(check_single, strict=strict)
        ret = mp_pool.map(check, tests)

    idx_failures = [i for i, x in enumerate(ret) if not x]
    is_failure = len(idx_failures) > 0

    if idx_failures:
        click.secho(f"Failures: {idx_failures}", fg="red")
    else:
        click.secho(f"Tests succeeded.", fg="green")

    send_notification("regression", is_failed=is_failure, idx_failures=idx_failures)


def create_regression_tests(num_tests, num_procs=1, write_out=False):
    """Create a regression vault.

    Parameters
    ----------
    num_test : int
        How many tests are in the vault.
    num_procs : int
        Number of processes. Default 1.
    write_out : bool
        If True, regression tests are stored to disk, replacing any existing regression
        tests. Be careful with this. Default False.

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
    params, options, exp_val = tests[idx]

    df = simulate_truncated_data(params, options)

    crit_func = rp.get_crit_func(params, options, df)

    crit_val = crit_func(params)

    np.testing.assert_almost_equal(crit_val, exp_val, decimal=DECIMALS)


def check_single(test, strict=False):
    """Check a single test."""
    params, option_spec, exp_val = test

    df = simulate_truncated_data(params, option_spec)

    crit_func = rp.get_crit_func(params, option_spec, df)

    est_val = crit_func(params)
    is_success = np.isclose(est_val, exp_val, rtol=TOL, atol=TOL)

    if strict is True:
        assert is_success, "Failed regression test."

    return is_success


def create_single(idx):
    """Create a single test."""
    np.random.seed(idx)
    params, options = generate_random_model()
    df = simulate_truncated_data(params, options)

    crit_func = rp.get_crit_func(params, options, df)

    crit_val = crit_func(params)

    if not isinstance(crit_val, float):
        raise AssertionError(" ... value of criterion function too large.")
    os.chdir("..")

    return params, options, crit_val


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """CLI manager for regression tests."""
    pass


@cli.command()
@click.option("-n", "--number", default=None, help="Number of regression tests.")
@click.option("-s", "--strict", default=False, help="Immediate termination on failure.")
@click.option("-p", "--parallel", default=1, help="Number of parallel tests.")
def run(ctx, number, strict, parallel):
    """Run a number of regression tests."""
    run_regression_tests(num_tests=number, strict=strict, num_procs=parallel)


@cli.command()
@click.option("-n", "--number", required=True, help="Number of single regression test.")
def investigate(number):
    """Investigate a single regression test."""
    investigate_regression_test(number)


@cli.command()
@click.option("-n", "--number", required=True, help="Number of regression tests.")
@click.option("-p", "--parallel", default=1, help="Number of parallel tests.")
def create(number, parallel):
    """Create a new collection of regression tests."""
    create_regression_tests(num_tests=number, procs=parallel, write_out=True)


if __name__ == "__main__":
    cli()
