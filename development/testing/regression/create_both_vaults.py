#!/usr/bin/env python
import subprocess
import argparse

from config import python2_exec
from config import python3_exec

# path to the script that must run under the virtualenv
script_file = 'create_vault.py'


def dist_input_arguments(parser):
    """ Check input for script.
    """
    num_tests = parser.parse_args().num_tests
    if num_tests is not None:
        assert isinstance(num_tests, int)
        assert (0 < num_tests < 500)
    return num_tests


def run(num_tests):
    for python_bin in [python2_exec, python3_exec]:
        cmd = [python_bin, script_file]
        if num_tests is not None:
            cmd += ['--tests', str(num_tests)]
        subprocess.check_call(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vaults',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--tests', action='store', dest='num_tests',
        default=None, help='number of tests to create', type=int)

    num_tests = dist_input_arguments(parser)

    run(num_tests)
