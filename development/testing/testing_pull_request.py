"""Run a series of tests that are required for any pull request to be merged."""
import socket

import click
from regression import run_regression_tests

import respy as rp


def run_pull_request_tests():
    # Determine whether to do a long or short run.
    is_short_run = socket.gethostname() in ["abacus", "socrates"]

    click.secho("Starting pytest", fg="green")
    rp.test()
    click.secho("Stopping pytest", fg="green")

    num_tests = 50 if is_short_run else 1000

    click.secho("Starting regression test.", fg="green")
    run_regression_tests(num_tests=num_tests, num_procs=1, strict=True)
    click.secho("Stopping regression test.", fg="green")


def main():
    """Run tests for validating pull request."""
    run_pull_request_tests()


if __name__ == "__main__":
    main()
