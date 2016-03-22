#!/usr/bin/env python
""" I will now try to run some estimations.
"""


# standard library
import numpy as np
from scipy.optimize import minimize
import sys
import os

# ROOT DIRECTORY
sys.path.insert(0, os.environ['ROBUPY'])

from robupy import simulate, read, solve, process, estimate

robupy_obj = read('test.robupy.ini')

# First, I simulate a dataset.
solve(robupy_obj)

#simulate(robupy_obj)

#data_frame = process(robupy_obj)

#estimate(robupy_obj, data_frame)
