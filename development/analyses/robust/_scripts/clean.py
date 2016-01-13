#!/usr/bin/env python
""" This script cleans the directories.
"""

# standard library
import shutil
import glob
import os

# module wide variables
SAVE_FILES = ['modules', 'clean', 'create', 'acropolis.pbs',
                  'auxiliary.py', 'clean.py', 'create.py', 'update',
                  'update.py', 'graphs.py', 'graphs', 'aggregate',
              'aggregate.py']

''' Auxiliary functions
'''


def cleanup(save_files=SAVE_FILES):
    """ Cleanup during development.
    """

    for name in glob.glob('*'):

        if name in save_files:
            pass
        else:
            remove(name)


def remove(names):
    """ Remove files or directories.
    """
    if not isinstance(names, list):
        names = [names]

    for name in names:
        try:
            os.remove(name)
        except IsADirectoryError:
            shutil.rmtree(name)

''' Main
'''
if __name__ =='__main__':

    root_dir = os.getcwd()

    for subdir, _, _ in os.walk('.'):

        os.chdir(subdir), cleanup(SAVE_FILES), os.chdir(root_dir)
