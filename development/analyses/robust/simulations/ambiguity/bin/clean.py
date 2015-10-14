#!/usr/bin/env python
""" This script cleans the directory.
"""

# standard library
import argparse
import shutil
import glob
import os

# Check for Python 3
import sys
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')
    
''' Functions
'''


def remove(file_):
    """ Remove directory.
    """
    try:
        os.unlink(file_)
    except IsADirectoryError:
        shutil.rmtree(file_)    


def cleanup():
    """ Clean directory.
    """

    files = glob.glob('*')

    for file_ in files:

        if file_ in ['graphs', 'acropolis.pbs', 'model.robupy.ini']:
            continue

        if file_ in ['bin', 'clean', 'create', 'update']:
            continue

        remove(file_)

    # Cleanup of script directory
    os.chdir('bin')
    try:
        shutil.rmtree('__pycache__')
    except:
        pass

    for file_ in glob.glob('*'):
        if '.pyc' in file_:
            remove(file_)
        continue

    os.chdir('../')


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Clean directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Run function
    cleanup()
