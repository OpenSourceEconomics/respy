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
# ROOT DIRECTORY
ROOT_DIR = os.environ['ROBUPY']
ROOT_DIR = ROOT_DIR + '/robupy/tests'

sys.path.insert(0, ROOT_DIR)

# testing codes

from robupy import simulate, read, solve, process, evaluate


robupy_obj = read('test.robupy.ini')

# First, I simulate a dataset.
robupy_obj = solve(robupy_obj)

val = robupy_obj.get_attr('periods_emax')[0, 0]
np.testing.assert_allclose(1.4963828613937988, val)

simulate(robupy_obj)
val = evaluate(robupy_obj, process(robupy_obj))
np.testing.assert_allclose(2.992618550039753, val)

