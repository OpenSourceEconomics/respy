#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('./waf distclean; ./waf configure build ') == 0
    os.chdir(cwd)




import os
import sys
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



respy_obj = RespyCls('model.respy.ini')
simulate(respy_obj)


for num_procs in [1, 5]:

	respy_obj.unlock()
	respy_obj.set_attr('num_procs', num_procs)
	respy_obj.set_attr('is_parallel', (num_procs > 1))
	respy_obj.lock()

	x, crit_val = estimate(respy_obj)

	print(num_procs, crit_val)
# print('working PYTHON')
# respy_obj = RespyCls('model.respy.ini')
# #respy_obj.attr['version'] = 'PYTHON'
# #respy_obj.attr['optimizer_used'] = 'SCIPY-POWELL'
# import time
# start = time.time()
#
# x, crit_val = estimate(respy_obj)
# print(crit_val, 'ONLY WORKING WIT MAXFUN 0')
#
# np.testing.assert_almost_equal(crit_val, 0.66798246030058295)
