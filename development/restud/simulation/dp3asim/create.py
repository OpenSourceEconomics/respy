#!/usr/bin/env python
""" Runs the original source code for the three models specified in the
papers.
"""

# standard library
import os
import shutil

# Compile executable
os.system(' gfortran -o dp3asim dp3asim.f95 > /dev/null 2>&1 ')

# Ensure fresh start
for which in ['one', 'two', 'three']:

    # Clean directories
    try:
        shutil.rmtree('data_' + which)
    except OSError:
        pass

    # Clean model specification file
    try:
        os.remove('in.txt')
    except OSError:
        pass

# Iterate over model specifications
for i, which in enumerate(['one', 'two', 'three']):

    # Storage for results
    os.mkdir('data_' + which)

    # Link to model specification
    os.symlink('in' + str(i + 1) + '.txt', 'in.txt')

    # Solve and simulate model
    os.system('./dp3asim  > /dev/null 2>&1 ')

    # Clean
    os.remove('in.txt')

    # Move material for inspection
    for file_ in ['ftest.txt', 'otest.txt']:
        shutil.move(file_, 'data_' + which + '/')

# Final cleanup
for file_ in ['dp3asim', 'imsl_replacements.mod']:
    os.remove(file_)