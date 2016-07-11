#!/usr/bin/env python
import subprocess
import argparse
import sys
import os

# Required for PYTHON2/3 portability
sys.path.insert(0, 'modules')

from auxiliary import dist_input_arguments
from auxiliary import write_request

sys.path.insert(0, '../../modules')
from config import python2_exec
from config import python3_exec


def run(request, num_tests, version):

    write_request(num_tests)

    if request == 'check':
        script_files = ['modules/check_vault.py']
    elif request == 'create':
        script_files = ['modules/create_vault.py']
    else:
        raise AssertionError

    if version is None:
        python_bins = [python2_exec, python3_exec]
    elif version == 2:
        python_bins = [python2_exec]
    elif version == 3:
        python_bins = [python3_exec]
    else:
        raise AssertionError

    for script_file in script_files:
        for python_bin in python_bins:
            cmd = [python_bin, script_file]
            subprocess.check_call(cmd)

    os.unlink('request.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create or check both vaults',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--tests', action='store', dest='num_tests',
        default=500, help='number of tests to process', type=int)

    parser.add_argument('--request', action='store', dest='request',
        help='task to perform', type=str, required=True,
        choices=['create', 'check'])

    parser.add_argument('--version', action='store', dest='version',
        help='Python version', type=int, required=False, default=None)

    request, num_tests, version = dist_input_arguments(parser)

    run(request, num_tests, version)
