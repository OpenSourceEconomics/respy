#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('git clean -d -f; ./waf configure build --debug') \
           == 0
    os.chdir(cwd)




import shutil

import time


from respy.python.shared.shared_auxiliary import print_init_dict

import numpy as np
from respy.python.solve.solve_ambiguity import criterion_ambiguity, \
    get_worst_case, construct_emax_ambiguity


from respy import RespyCls
from respy import simulate
from respy import estimate

from codes.auxiliary import simulate_observed
from codes.auxiliary import write_draws

from codes.random_init import generate_init
from respy.python.shared.shared_auxiliary import dist_class_attributes
#
# np.random.seed(123)
# respy_obj = RespyCls('model.respy.ini')
#
# base_crit = None
# for version in [1]:
# #    version = 'procs' + str(num_procs)
# #    if os.path.exists(version):
# #        shutil.rmtree(version)
#
#     #os.mkdir(version)
#     #os.chdir(version)
#     #shutil.copy('../draws.txt', 'draws.txt')
#  #   respy_obj.reset()
#  #   respy_obj.unlock()
#  #   respy_obj.attr['num_procs'] = num_procs
#  #   respy_obj.lock()
#     respy_obj = simulate_observed(respy_obj)
#     _, crit = estimate(respy_obj)
#     if base_crit is None:
#        base_crit = crit
#     np.testing.assert_almost_equal(crit, base_crit)
#
# #   os.chdir('../')
#     print(crit)
for _ in range(200):
    max_draws = np.random.randint(10, 100)

    # Generate random initialization file
    constr = dict()
    constr['flag_ambiguity'] = True
    constr['flag_parallelism'] = False
    constr['max_draws'] = max_draws
    constr['flag_interpolation'] = False
    constr['maxfun'] = 0

    #os.system('git clean -d -f')

    # Generate random initialization file
    init_dict = generate_init(constr)

    for version in ['FORTRAN', 'PYTHON']:
        print(version)
        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        # Iterate over alternative implementations
        base_x, base_val = None, None


        if os.path.exists(version):
            shutil.rmtree(version)

        os.mkdir(version)
        os.chdir(version)


        # Simulate a dataset
        simulate_observed(respy_obj)

        num_periods = respy_obj.get_attr('num_periods')
        write_draws(num_periods, max_draws)


        respy_obj.unlock()

        respy_obj.set_attr('version', version)

#        shutil.copy('../draws.txt', '.')
#        respy_obj.attr['file_est'] = '../data.respy.dat'

        respy_obj.lock()

        x, val = estimate(respy_obj)

        os.chdir('../')

        print(val)
        # Check for the returned parameters.
        if base_x is None:
            base_x = x
        np.testing.assert_allclose(base_x, x)

        # Check for the value of the criterion function.
        if base_val is None:
            base_val = val
        np.testing.assert_allclose(base_val, val)