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



from respy.python.evaluate.evaluate_auxiliary import check_input
from respy.python.evaluate.evaluate_auxiliary import check_output

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import create_draws

from respy import simulate, RespyCls, estimate
import numpy as np

import pickle as pkl


#sys.path.insert(0, '/home/peisenha/restudToolbox/package/respy/tests/resources')
respy_obj = RespyCls('model.respy.ini')
#respy_obj = simulate(respy_obj)
x, crit_val = estimate(respy_obj)

np.testing.assert_equal(crit_val, 4.1093e-11)
#print(respy_obj.get_attr('periods_emax')[0, 0])
