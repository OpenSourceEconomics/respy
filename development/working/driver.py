#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
if True:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('./waf distclean; ./waf configure build --debug') == 0
    os.chdir(cwd)




# ROOT DIRECTORY
# standard library
import os
import sys
import shutil

import time
# ROOT DIRECTORY

# project library
from respy.python.evaluate.evaluate_python import pyth_evaluate


from respy.python.evaluate.evaluate_auxiliary import check_input
from respy.python.evaluate.evaluate_auxiliary import check_output

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import create_draws

from respy import simulate, RespyCls, estimate
from respy.solve import solve
import numpy as np

import pickle as pkl
sys.path.insert(0, '/home/peisenha/Dropbox/business/office/workspace/software/repositories/organizations/restudToolbox/package/respy/tests')
from codes.auxiliary import write_draws



respy_obj = RespyCls('model.respy.ini')
simulate(respy_obj)
#x, crit_val = estimate(respy_obj)
print('working PYTHON')
respy_obj = RespyCls('model.respy.ini')
#respy_obj.attr['version'] = 'PYTHON'
#respy_obj.attr['optimizer_used'] = 'SCIPY-POWELL'
import time
start = time.time()

x, crit_val = estimate(respy_obj)
print(crit_val, 'ONLY WORKING WIT MAXFUN 0')

np.testing.assert_almost_equal(crit_val, 0.66798246030058295)