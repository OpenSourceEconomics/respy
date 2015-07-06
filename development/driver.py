#!/usr/bin/env python

""" This module is used for the development setup.
"""

# project library
import time
import sys
import os

sys.path.insert(0, os.environ['ROBUPY'])

# project library
from robupy import *

# Run workflowR
robupy_obj = read('model.robupy.ini')

print(' Starting to solve the model ...')

start_time = time.time()

robupy_obj = solve(robupy_obj)

exit_time = time.time()

print('    Duration: ', (exit_time - start_time)/60)

print('\n Starting to simulate ...')

simulate(robupy_obj)
