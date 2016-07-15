#!/usr/bin/env python
import subprocess
import argparse
import sys
import os

sys.path.insert(0, '../_modules')
from auxiliary_regression import write_request
from auxiliary_shared import send_notification
from config import python2_exec
from config import python3_exec


def run(args):

    write_request(args.num_tests)

    if args.request == 'check':
        script_files = ['modules/check_vault.py']
    elif args.request == 'create':
        script_files = ['modules/create_vault.py']
    else:
        raise AssertionError

    if args.version is None:
        python_bins = [python2_exec, python3_exec]
    elif args.version == 2:
        python_bins = [python2_exec]
    elif args.version == 3:
        python_bins = [python3_exec]
    else:
        raise AssertionError

    for script_file in script_files:
        for python_bin in python_bins:
            cmd = [python_bin, script_file]
            subprocess.check_call(cmd)

    os.unlink('request.txt')

    # This allows to call this test from another script, that runs other
    # tests as well.
    if not args.is_background:
        send_notification('regression')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create or check both vaults',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--tests', action='store', dest='num_tests',
        default=500, help='number of tests to process', type=int)

    parser.add_argument('--request', action='store', dest='request',
        help='task to perform', type=str, required=True,
        choices=['create', 'check'])

    parser.add_argument('--version', action='store', dest='version',
        help='Python version', type=int, required=False, default=None,
        choices=[2, 3])

    parser.add_argument('--background', action='store_true',
        dest='is_background', default=False, help='background process')

    run(parser.parse_args())
