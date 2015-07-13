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

import numpy as np
import robupy.fort.fortran_functions as fort

num_cols = 4
A = np.identity(num_cols)




np.random.seed(2)
for _ in range(100000):

    x = np.random.random(2)

    matrix = (np.random.multivariate_normal(np.zeros(4),
                                np.identity(4), (4)))

    cov = np.dot(matrix, matrix.T)


    level = np.random.random(1)[0]
    print('\n\n NEW TASK')
    print(cov)
    py = np.linalg.inv(cov)
    f90 = fort.inverse(cov, 4)

    print(py - f90)

    np.testing.assert_allclose(py, f90, rtol = 1e-06)
    #assert (np.abs(py - f90) < 10e-6)