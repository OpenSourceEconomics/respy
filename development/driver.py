""" This module is used for the development setup.
"""

# project library
import sys
import os

sys.path.insert(0, os.environ['ROBUPY'])

#sys.path.insert(0, '/Users/peisenha/robustToolbox/development/tests/random
# /modules')

#from _random_init import _random_dict

# project library
from robupy import *

# Run workflowR
robupy_obj = read('model.robupy.ini')

robupy_obj = solve(robupy_obj)

simulate(robupy_obj)

# Cleanup
os.remove('data.robupy.dat')

os.remove('data.robupy.info')


if False:
    import numpy as np
    import scipy.linalg

    np.random.seed(345)
    num_draws = 1000
    num_dim = 2


    import numpy as np

    # Add mean

    #lower_triangular = (np.tril(np.random.multivariate_normal(np.zeros(
    # num_dim), np.identity(num_dim), (num_dim))))

    #lower_triangular = lower_triangular**2

    #print(lower_triangular)
    #print('\n' )
    if False:
        covs = np.array([[1.0, 0.2],
                         [0.2, 1.0]])
    else:
        covs = np.array([[1.0, 0.0],
                         [0.0, 1.0]])
    #print(covs)
    lower_triangular = np.linalg.cholesky(covs)

    print(np.dot(lower_triangular, lower_triangular.T))


    # Draw directly
    np.random.seed(123)
    direct = np.random.multivariate_normal(np.zeros(num_dim), covs, (num_draws))

    # Draw indirectly
    np.random.seed(123)
    eps = np.random.multivariate_normal(np.zeros(num_dim), np.identity(num_dim),
                                        (num_draws))

    indirect = np.dot(lower_triangular, eps.T).T
