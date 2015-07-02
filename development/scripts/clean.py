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

        for filename in fnmatch.filter(dir_, '.vagrant'):
            matches.append(os.path.join(root, filename))

        for file_types in ['*.aux', '*.log', '*.pyc', '*.so',  '*tar',
                           '*.bbl', '*.blg', '*.out',
                           '*.zip', '.waf*', '*lock*', '*.mod', '*.a',
                           '*.snm', '*.toc', '*.nav', 'main.pdf',
                           '__pycache__', '.DS_Store', '*.pyo', '*.profile',
                           'doc.pdf', 'dataset.txt', 'results.txt',
                           'simulated.txt', '.coverage', '.vagrant']:

            for filename in fnmatch.filter(file_names, file_types):

                matches.append(os.path.join(root, filename))

    for files in matches:

        remove(files)

    for root, dir_, file_names in os.walk('.'):

        if 'ipynb_checkpoints' in root:

            shutil.rmtree(root)

        if 'lecture_files' in root:

            shutil.rmtree(root)

""" Main algorithm.
"""
remove_nuisances()

