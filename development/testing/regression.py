"""Create, run or investigate regression checks."""
import pickle
import socket

import click
import numpy as np

from respy.config import TEST_RESOURCES_DIR
from respy.config import TOL_REGRESSION_TESTS
from respy.tests.random_model import generate_random_model
from respy.tests.test_regression import compute_log_likelihood
from respy.tests.test_regression import load_regression_tests

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


def run_regression_tests(n_tests, strict, notification):
    """Run regression tests.

    Parameters
    ----------
    n_tests : int
        Number of tests to run. If None, all are run.
    strict : bool, default False
        Early failure on error.
    notification : bool, default True
        Send notification with test report.

    """
    tests = load_regression_tests()
    tests = tests[: n_tests + 1]

    results = [_check_single(test, strict) for test in tests]
    idx_failures = [i for i, x in enumerate(results) if not x]

    if idx_failures:
        click.secho(f"Failures: {idx_failures}", fg="red")
    else:
        click.secho(f"Tests succeeded.", fg="green")

    subject, message = _prepare_message(idx_failures)

    if notification:
        from development.testing.notifications import send_notification

        send_notification(subject, message)


def create_regression_tests(n_tests, save):
    """Create a regression vault.

    Parameters
    ----------
    n_tests : int
        How many tests are in the vault.
    save : bool, default True
        Flag for saving new tests to disk.

    """
    tests = [_create_single(i) for i in range(n_tests)]

    if save:
        with open(TEST_RESOURCES_DIR / "regression_vault.pickle", "wb") as p:
            pickle.dump(tests, p)


def investigate_regression_test(idx):
    """Investigate regression tests."""
    tests = load_regression_tests()
    params, options, exp_val = tests[idx]

    crit_val = compute_log_likelihood(params, options)

    assert np.isclose(
        crit_val, exp_val, rtol=TOL_REGRESSION_TESTS, atol=TOL_REGRESSION_TESTS
    )


def _check_single(test, strict):
    """Check a single test."""
    params, options, exp_val = test

    try:
        crit_val = compute_log_likelihood(params, options)
        is_success = np.isclose(
            crit_val, exp_val, rtol=TOL_REGRESSION_TESTS, atol=TOL_REGRESSION_TESTS
        )
    except Exception:
        is_success = False

    if strict is True:
        assert is_success, "Failed regression test."

    return is_success


def _create_single(idx):
    """Create a single test."""
    np.random.seed(idx)

    params, options = generate_random_model()

    crit_val = compute_log_likelihood(params, options)

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
@click.option("--notification/--no-notification", default=True, help="Send report.")
def run(number_of_tests, strict, notification):
    """Run a number of regression tests."""
    run_regression_tests(
        n_tests=number_of_tests, strict=strict, notification=notification
    )


@cli.command()
@click.argument("number_of_test", type=int)
def investigate(number_of_test):
    """Investigate a single regression test."""
    investigate_regression_test(number_of_test)


@cli.command()
@click.argument("number_of_tests", type=int)
@click.option("--save/--no-save", default=True, help="Saves new tests on disk.")
def create(number_of_tests, save):
    """Create a new collection of regression tests."""
    create_regression_tests(n_tests=number_of_tests, save=save)


if __name__ == "__main__":
    cli()
