#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
import numpy as np

import scipy
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# RobuPy library
from robupy import *

from robupy.auxiliary import opt_get_model_parameters
from robupy.auxiliary import opt_get_optim_parameters
from robupy.auxiliary import distribute_model_paras

# testing battery
from modules.auxiliary import compile_package

# Read in baseline initialization file
compile_package('--fortran --debug', False)

np.random.seed(123)

import robupy.python.f2py.f2py_debug as fort

# Draw random requests for testing purposes.
num_draws = np.random.random_integers(2, 1000)
dim = np.random.random_integers(1, 6)
mean = np.random.uniform(-0.5, 0.5, (dim))

matrix = (np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim))
cov = np.dot(matrix, matrix.T)

# Singular Value Decomposition
py = scipy.linalg.svd(matrix)[0]
f90 = fort.wrapper_svd(matrix, dim)[0]

for i in range(3):
    np.testing.assert_allclose(py[i], f90[i], rtol=1e-05, atol=1e-06)

''' F
'''
robupy_obj = read('test.robupy.ini')

solve(robupy_obj)


