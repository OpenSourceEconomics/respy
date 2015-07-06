#!/usr/bin/env python
""" Cleanup of directories.
"""

# standard library
import fnmatch
import shutil
import os

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

        for filename in fnmatch.filter(dir_, '__pycache__'):
            matches.append(os.path.join(root, filename))

        for file_types in ['*.robupy.*']:

            for filename in fnmatch.filter(file_names, file_types):

                matches.append(os.path.join(root, filename))

    for file_ in matches:

        if 'ini' in file_: continue

        remove(file_)


""" Main algorithm.
"""
remove_nuisances()

