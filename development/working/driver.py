#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('git clean -d -f; ./waf configure build --debug ') == 0
    os.chdir(cwd)




import shutil

import time

from respy.python.evaluate.evaluate_python import pyth_evaluate


from respy.python.evaluate.evaluate_auxiliary import check_input
from respy.python.evaluate.evaluate_auxiliary import check_output

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import create_draws

from respy import simulate, RespyCls, estimate
import numpy as np

import pickle as pkl


sys.path.insert(0, '/home/peisenha/restudToolbox/package/respy/tests/resources')



from f2py_interface import wrapper_kl_divergence
from respy.python.solve.solve_ambiguity import kl_divergence

for i in range(10000):
    print(i)
    num_dims = np.random.randint(1, 5)

    old_mean = np.random.uniform(size=num_dims)
    new_mean = np.random.uniform(size=num_dims)

    cov = np.random.random((num_dims, num_dims))
    old_cov = np.matmul(cov.T, cov)

    cov = np.random.random((num_dims, num_dims))
    new_cov = np.matmul(cov.T, cov)

    # Stabilization for inverse.
    np.fill_diagonal(new_cov, new_cov.diagonal() * 5)
    np.fill_diagonal(old_cov, old_cov.diagonal() * 5)


    fort = wrapper_kl_divergence(old_mean, old_cov, new_mean, new_cov)
    pyth = kl_divergence(old_mean, old_cov, new_mean, new_cov)


    np.testing.assert_almost_equal(fort, pyth)