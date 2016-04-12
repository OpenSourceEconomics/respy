#!/usr/bin/env python

""" Cleanup of directories.
"""

# standard library
import fnmatch
import shutil
import os

# module-wide variables
PROJECT_DIR = os.environ['ROBUPY']

""" Auxiliary functions.
"""


def remove(path):
    """ Remove path, where path can be either a directory or a file. The
        appropriate function is selected. Note, however, that if an
        OSError occurs, the function will just path.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)

    if os.path.isfile(path):
        os.remove(path)


def remove_nuisances():
    """ Remove nuisance files from the directory tree.
    """

    matches = []

    for root, dir_, file_names in os.walk('.'):

        # Cleanup of directories
        for name in ['__pycache__', '.cache', 'build', '.eggs',
            'robupy.egg-info']:
            for filename in fnmatch.filter(dir_, name):
                matches.append(os.path.join(root, filename))

        for file_types in ['*.robupy.*', '.lock-waf_linux_build',
                           '*.so', '.coverage', '*.txt', '.write_out',
                           'dp3asim', '*.o', '*.tmp', '*.pyc']:

            for filename in fnmatch.filter(file_names, file_types):
                matches.append(os.path.join(root, filename))

    for file_ in matches:

        if 'ini' in file_:
            continue

        if 'opt' in file_:
            continue

        if 'restud/simulation' in file_:
            continue

        if 'restud/codes' in file_:
            continue

        if 'requirements' in file_:
            continue

        if 'rslts' in file_:
            continue

        remove(file_)

    # Iterate for directories and
    for root, dir_, file_names in os.walk('.'):

        if '.bld' in root:

            shutil.rmtree(root)

        if '.waf' in root:

            shutil.rmtree(root)

        if 'ipynb_checkpoints' in root:

            shutil.rmtree(root)

        if 'include' in root:

            shutil.rmtree(root)

        if 'lib' in root:

            shutil.rmtree(root)

""" Main algorithm.
"""
remove_nuisances()

os.chdir(PROJECT_DIR + '/robupy')

os.system('./waf distclean')

remove('../development/testing/test.robupy.ini')
