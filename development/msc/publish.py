#!/usr/bin/env python
""" This script publishes the package on PYPI.
"""

# standard library
import argparse
import shutil
import shlex
import glob
import os

# automate minor increment

def publish(version):
    """ Publish to package repository.
    """

    pass


''' Execution of module as script
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Publish ROBUPY on PYPI.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--version', action='store', dest='version',
                        default=None, help='version to publish')

    args = parser.parse_args()

    ROOT_DIR = os.getcwd()

    # Cleanup
    os.chdir('package/robupy')
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
    os.system('python setup.py sdist upload -r pypi')
    os.chdir(ROOT_DIR)
