#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
from scipy.stats import norm

import numpy as np

import scipy
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests')
sys.path.insert(0, os.environ['ROBUPY'])

# RobuPy library
from robupy import simulate
from robupy import evaluate
from robupy import read
from robupy import solve


from material.auxiliary import write_disturbances


print('not recompiling')
#build_robupy_package(False)

np.random.seed(123)

robupy_obj = read('test.robupy.ini')

num_periods = robupy_obj.get_attr('num_periods')
max_draws = 1000
write_disturbances(num_periods, max_draws)


robupy_obj = solve(robupy_obj)

data_frame = simulate(robupy_obj)

print(evaluate(robupy_obj, data_frame))