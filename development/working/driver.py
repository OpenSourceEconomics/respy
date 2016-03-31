#!/usr/bin/env python
""" I will now try to run some estimations.
"""


# standard library
import os
import sys

# ROOT DIRECTORY
sys.path.insert(0, os.environ['ROBUPY'])
# ROOT DIRECTORY
ROOT_DIR = os.environ['ROBUPY']
ROOT_DIR = ROOT_DIR + '/robupy/tests'

sys.path.insert(0, ROOT_DIR)

# testing codes

from robupy import simulate, read, solve

robupy_obj = read('model.robupy.ini')

print('starting to simulate')
data_frame = simulate(robupy_obj)


# Indicator for working at time t in Occupation 1
t = 1

is_working = (data_frame[2] == 1) & (data_frame[1] == t)
wages = data_frame[is_working].ix[:,3]

