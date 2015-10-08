#!/usr/bin/env python
""" Clean directory.
"""

# standard library
import shutil
import os

# module-wide variables
DIRECTORIES = ['__pycache__', 'rslts', 'modules/__pycache__']

# module-wide variables
HOME = os.path.dirname(os.path.realpath(__file__))

# Setup
os.chdir(HOME)

# Remove directories
for directory in DIRECTORIES:
    try:
        shutil.rmtree(directory)
    except OSError:
        pass

