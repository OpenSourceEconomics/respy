#!/usr/bin/env python
""" Runs the original source code for the three models specified in the
papers.
"""


import os
import shutil


os.system(' gfortran -o dp3asim dp3asim.f95')

# Prepare directories
for which in ['one', 'two', 'three']:
    shutil.rmtree('data_' + which)

for i, which in enumerate(['one', 'two', 'three']):

    try:
        os.mkdir('data_' + which)
    except OSError:
        pass

    try:
        os.symlink('in' + str(i + 1) + '.txt', 'in.txt')
    except OSError:
        pass

    os.system('./dp3asim')

    try:
        os.remove('in.txt')
    except OSError:
        pass

    # Move material for inspection
    for file_ in ['ftest.txt', 'otest.txt']:

        shutil.move(file_, 'data_' + which + '/')


# Final cleanup
for file_ in ['dp3asim', 'imsl_replacements.mod']:
    os.remove(file_)