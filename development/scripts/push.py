#!/usr/bin/env python
""" Push to all from remote repositories.
"""

# standard library
import argparse
import os

# module-wide variables
REPOS = ['development', 'documentation', 'package']

REPOS += ['robustToolbox.github.io']


''' Functions
'''
def push():
    """ Push all remote repositories.
    """
    for repo in REPOS:

        # Check if checked out
        if repo not in os.listdir('.'):
            continue

        print('\n.. pushing ' + repo)

        os.chdir(repo)

        os.system("git clean -f") 

        os.system("git commit -a -m'committing'") 

        os.system("git push")

        os.chdir('../')

    print('')

''' Execution of module as script
'''
if __name__ == '__main__':

    parser  = argparse.ArgumentParser(description = \
        'Push to all remote repositories.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    push()
