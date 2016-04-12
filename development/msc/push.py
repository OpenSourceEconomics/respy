#!/usr/bin/env python
""" Push to all from remote repositories.
"""

# standard library
import argparse
import os

# module-wide variables
HOME = os.getcwd()

REPOS = ['documentation/sources', 'package']

REPOS += ['documentation/robustToolbox.github.io']

''' Functions
'''


def push():
    """ Push all remote repositories.
    """
    for repo in REPOS:

        os.chdir(repo)

        if os.path.exists('.git'):
            print('\n.. pushing ' + repo)
            os.system("git clean -f")
            os.system("git commit -a -m'committing'")
            os.system("git push")
            os.chdir('../')

        os.chdir(HOME)

    print('')

''' Execution of module as script
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Push to all remote repositories.', formatter_class=
        argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    push()
