#!/usr/bin/env python
""" Pull all from remote repositories.
"""

# standard library
import argparse
import os

# module-wide variables
REPOS = ['development', 'package']

''' Functions
'''
def update():
    """ Update all remote repositories.
    """
    for repo in REPOS:

        # Check if checked out
        if repo not in os.listdir('.'):
            continue

        print('\n.. updating ' + repo)

        os.chdir(repo)

        os.system('git pull')

        os.chdir('../')

    print('')

''' Execution of module as script
'''
if __name__ == '__main__':

    parser  = argparse.ArgumentParser(description = \
        'Update all remote repositories.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    update()
