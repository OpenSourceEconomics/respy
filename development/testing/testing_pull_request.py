"""Run a series of tests that are required for any pull request to be merged."""
import socket

import click
from click.testing import CliRunner
from testing.regression import run

import respy as rp


def run_pull_request_tests():
    is_short_run = socket.gethostname() in ["abacus", "socrates"]

    click.secho("Starting pytest", fg="green")
    rp.test()
    click.secho("Stopping pytest", fg="green")

    n_tests = 50 if is_short_run else 1000

    runner = CliRunner()

    click.secho("Starting regression test.", fg="green")
    runner.invoke(run, [str(n_tests), "--strict"])
    click.secho("Stopping regression test.", fg="green")


def main():
    """Run tests for validating pull request."""
    run_pull_request_tests()


if __name__ == "__main__":
    main()
