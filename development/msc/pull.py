#!/usr/bin/env python
""" Pull all from remote repositories.
"""

# standard library
import argparse
import os

# module-wide variables
HOME = os.getcwd()

REPOS = ['documentation/sources', 'package', 'analysis']

REPOS += ['documentation/robustToolbox.github.io']


''' Functions
'''


def update():
    """ Update all remote repositories.
    """
    for repo in REPOS:

        os.chdir(repo)

        if os.path.exists('.git'):
            print('\n.. updating ' + repo)
            os.system('git pull')

        os.chdir(HOME)

    print('')

''' Execution of module as script
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Update all remote repositories.', formatter_class=
        argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    update()
