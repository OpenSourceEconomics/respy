#!/usr/bin/env python
""" Starting of Monte Carlo specifications.
"""

# standard library
import glob
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'])
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')

# Import function to that a fast version of the toolbox is available.
from modules.auxiliary import compile_package

# Compile fast version of ROBUPY package
compile_package('--fortran', True)

# Start model solution and simulation
for dir_ in ['one', 'two', 'three']:

    os.chdir('data_' + dir_)

    os.system('robupy-solve')

    # Cleanup
    files = glob.glob('logging.*')
    for file_ in files:
        os.unlink(file_)

    os.chdir('..')
