""" This script allows to update the regression tests."""
import argparse
import shutil
import subprocess
import sys

from development.modules.auxiliary_shared import cleanup

PYTHON_EXEC = sys.executable


def run(num_procs, num_tests, is_check):
    # Initially we want to make sure that all the previous tests are running properly.
    if is_check:
        cleanup()

        # This scratch file indicates that the required modification is done properly.
        open(".old.respy.scratch", "w").close()
        cmd = PYTHON_EXEC + " run_regression.py --request check "
        cmd += str(num_tests) + " --strict" + " --procs " + str(num_procs)
        subprocess.check_call(cmd, shell=True)

    # We create a new set of regression tests.
    cleanup()
    cmd = PYTHON_EXEC + " run_regression.py --request create " + str(num_tests)
    cmd += " --procs " + str(num_procs)
    subprocess.check_call(cmd, shell=True)

    # These are subsequently copied into the test resources of the package.
    shutil.copy("regression_vault.respy.json", "../../../respy/tests/resources")

    # Just to be sure, we immediately check them again. This might fail if the random
    # elements are not properly controlled for.
    cleanup()
    cmd = PYTHON_EXEC + " run_regression.py --request check " + str(num_tests)
    cmd += " --strict" + " --procs " + str(num_procs)
    subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Update regression vault.")

    parser.add_argument(
        "--procs",
        action="store",
        dest="num_procs",
        default=1,
        type=int,
        help="number of processors",
    )

    parser.add_argument(
        "--tests",
        action="store",
        dest="num_tests",
        required=True,
        type=int,
        help="number of tests",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        dest="is_check",
        default=False,
        help="check current vault",
    )

    args = parser.parse_args()

    num_procs = args.num_procs
    num_tests = args.num_tests
    is_check = args.is_check

    run(num_procs, num_tests, is_check)
