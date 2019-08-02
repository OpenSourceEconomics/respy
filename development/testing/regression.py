"""Create, run or investigate regression checks."""
import pickle
import socket
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


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _prepare_message(idx_failures):
    hostname = socket.gethostname()
    subject = " respy: Regression Testing"
    if idx_failures:
        message = (
            f"Failure during regression testing @{hostname} for test(s): "
            f"{idx_failures}."
        )
    else:
        message = f"Regression testing is completed on @{hostname}."

    return subject, message


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
        ret = mp_pool.map(check_single, tests, kwargs={"strict": strict})

    idx_failures = [i for i, x in enumerate(ret) if not x]

    if idx_failures:
        click.secho(f"Failures: {idx_failures}", fg="red")
    else:
        click.secho(f"Tests succeeded.", fg="green")

    subject, message = _prepare_message(idx_failures)
    send_notification(subject, message)


def create_regression_tests(num_tests, num_procs=1):
    """Create a regression vault.

    Parameters
    ----------
    num_test : int
        How many tests are in the vault.
    num_procs : int, default 1
        Number of processes.

    """
    if num_procs == 1:
        tests = []
        for idx in range(num_tests):
            tests.append(create_single(idx))
    else:
        with Pool(num_procs) as p:
            tests = p.map(create_single, range(num_tests))

    with open(TEST_RESOURCES_DIR / "regression_vault.pickle", "wb") as p:
        pickle.dump(tests, p)


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

    # the negative sign is because before using estimagic we
    # returned the negative log likelihood and I did not want to make
    # new regression test.
    is_success = np.isclose(est_val, -exp_val, rtol=TOL, atol=TOL)

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

    return params, options, crit_val


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """CLI manager for regression tests."""
    pass


@cli.command()
@click.argument("number_of_tests", type=int)
@click.option("--strict", is_flag=True, help="Immediate termination on failure.")
@click.option("-p", "--parallel", default=1, type=int, help="Number of parallel tests.")
def run(number_of_tests, strict, parallel):
    """Run a number of regression tests."""
    run_regression_tests(num_tests=number_of_tests, strict=strict, num_procs=parallel)


@cli.command()
@click.argument("number_of_test", type=int)
def investigate(number_of_test):
    """Investigate a single regression test."""
    investigate_regression_test(number_of_test)


@cli.command()
@click.argument("number_of_tests", type=int)
@click.option("-p", "--parallel", default=1, type=int, help="Number of parallel tests.")
def create(number_of_tests, parallel):
    """Create a new collection of regression tests."""
    create_regression_tests(num_tests=number_of_tests, num_procs=parallel)


if __name__ == "__main__":
    cli()
