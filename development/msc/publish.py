#!/usr/bin/env python
""" This script publishes the package on PYPI.
"""

# standard library
import argparse
import shutil
import glob
import os

# module-wide variables
ROOT_DIR = os.getcwd()

''' Auxiliary functions
'''


def distribute_inputs(parser_internal):
    """ Process input arguments.
    """
    # Parse arguments.
    args = parser_internal.parse_args()

    # Distribute arguments
    is_local_internal = args.local

    # Assertions
    assert (is_local_internal in [True, False])

    # Finishing
    return is_local_internal


def publish(is_local_internal):
    """ Publish to package repository.
    """

    # Cleanup
    os.chdir('package/respy')
    os.system('./waf distclean')
    os.chdir(ROOT_DIR)

    # Get current version
    os.chdir('package/dist')

    latest = '0'
    for package in glob.glob('*'):
        name = package.split('-')[-1].split('.tar')[0]
        if name > latest:
            latest = name

    # Construct label for update
    indicator = int(latest.split('.')[-1])
    update = latest.replace(str(indicator), str(indicator + 1))
    os.chdir(ROOT_DIR)

    # Update setup file
    os.chdir('package')

    with open('setup.py', 'r') as old_file:
        with open('setup_updated.py', 'w') as new_file:
            for line in old_file.readlines():
                if 'version' in line:
                    new_file.write('  version = "' + update + '" ,\n')
                else:
                    new_file.write(line)

    shutil.move('setup_updated.py', 'setup.py')

    os.chdir(ROOT_DIR)

    # Publish
    os.chdir('package')

    if is_local_internal:
        os.system('python setup.py sdist')
    else:
        os.system('python setup.py sdist upload -r pypi')

    os.chdir(ROOT_DIR)


''' Execution of module as script
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Publish ROBUPY on PYPI.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--version', action='store', dest='version',
                        default=None, help='version to publish')

    parser.add_argument('--local', action='store_true', dest='local',
                        default=False, help='local test')

    is_local = distribute_inputs(parser)

    publish(is_local)

