#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('./waf distclean; ./waf configure build '
                     '--debug --without_f2py --without_parallelism') == 0
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
from respy.scripts.scripts_compare import scripts_compare
from codes.auxiliary import simulate_observed
from codes.auxiliary import write_draws

from codes.random_init import generate_init
from respy.python.shared.shared_auxiliary import dist_class_attributes

#write_draws(5, 5000)
np.random.seed(123)
#print 'running with types'
respy_obj = RespyCls('model.respy.ini')
# This ensures that the experience effect is taken care of properly.
open('.restud.respy.scratch', 'w').close()

respy_obj, _ = simulate(respy_obj)
#respy_obj.write_out('test.respy.ini')
#respy_obj = RespyCls('test.respy.ini')
_, crit = estimate(respy_obj)
scripts_compare('model.respy.ini', True)
