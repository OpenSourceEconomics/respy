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

np.random.seed(123)
respy_obj = RespyCls('model.respy.ini')

base_crit = None
for num_procs in [1]:
#    version = 'procs' + str(num_procs)
#    if os.path.exists(version):
#        shutil.rmtree(version)

    #os.mkdir(version)
    #os.chdir(version)
    #shutil.copy('../draws.txt', 'draws.txt')
 #   respy_obj.reset()
 #   respy_obj.unlock()
 #   respy_obj.attr['num_procs'] = num_procs
 #   respy_obj.lock()
    respy_obj = simulate_observed(respy_obj)
    _, crit = estimate(respy_obj)
    if base_crit is None:
       base_crit = crit
    np.testing.assert_almost_equal(crit, base_crit)

#    os.chdir('../')
#    print(crit)
