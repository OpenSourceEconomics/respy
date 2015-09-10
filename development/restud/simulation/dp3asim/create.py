#!/usr/bin/env python
""" Runs the original source code for the three models specified in the
papers.
"""

# standard library
import os

# Compile executable
os.system(' gfortran -o dp3asim dp3asim.f95 > /dev/null 2>&1 ')

# Iterate over model specifications
for i, which in enumerate(['one', 'two', 'three']):

    # Storage for results
    os.chdir('data_' + which)

    os.symlink('../dp3asim', 'dp3asim')

    # Solve and simulate model
    os.system('./dp3asim > /dev/null 2>&1 ')

    # Clean
    os.remove('dp3asim')

    os.chdir('../')

# Final cleanup
for file_ in ['dp3asim', 'imsl_replacements.mod', 'pei_additions.mod']:
    os.remove(file_)