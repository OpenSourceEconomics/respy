#!/usr/bin/env python
""" Starting of Monte Carlo specifications.
"""

# standard library
import os

# Start model solution and simulation
for dir_ in ['one', 'two', 'three']:

    os.chdir('data_' + dir_)

    os.system('robupy-clean &')

    os.system('robupy-solve --simulate &')

    os.chdir('..')
