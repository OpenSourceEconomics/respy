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



def divergence(x, cov, level):
    """ Calculate the relevant Kullback-Leibler distance of evaluation points
        from center.
    """
    # Construct alternative distribution
    alt_mean = np.zeros(4)
    alt_mean[:2] = x
    alt_cov = cov

    # Construct baseline distribution
    old_mean, old_cov = np.array([0.0, 0.0, 0.0, 0.0]), cov

    # Calculate distance
    comp_a = np.trace(np.dot(np.linalg.inv(old_cov), alt_cov))


    comp_b = np.dot(np.dot(np.transpose(old_mean - alt_mean),
            np.linalg.inv(old_cov)), (old_mean - alt_mean))

    print(np.linalg.inv(old_cov))
    comp_c = np.log(np.linalg.det(alt_cov) / np.linalg.det(old_cov))


    rslt = 0.5 * (comp_a + comp_b - 4 + comp_c)

    # Finishing.
    return level - rslt

np.random.seed(2)
for _ in range(100000):

    x = np.random.random(2)

    matrix = (np.random.multivariate_normal(np.zeros(4),
                                np.identity(4), (4)))

    cov = np.dot(matrix, matrix.T)


    level = np.random.random(1)[0]
    print('\n\n NEW TASK')
    print(cov)
    py = divergence(x, cov, level)
    f90 = fort.divergence(x, cov, level)

    print(py - f90)

    #np.testing.assert_allclose(py, f90, rtol = 1e-06)
    assert (np.abs(py - f90) < 10e-6)