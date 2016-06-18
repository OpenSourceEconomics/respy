#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
if False:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('./waf distclean; ./waf configure build --debug ') == 0
    os.chdir(cwd)




# ROOT DIRECTORY
# standard library
import os
import sys
import shutil

import time
# ROOT DIRECTORY
from respy.python.estimate.estimate_auxiliary import dist_optim_paras

# project library
from respy.python.evaluate.evaluate_python import pyth_evaluate


from respy.python.evaluate.evaluate_auxiliary import check_input
from respy.python.evaluate.evaluate_auxiliary import check_output

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import create_draws

from respy import simulate, RespyCls, estimate
from respy.solve import solve
import numpy as np

import pickle as pkl
sys.path.insert(0, '/home/peisenha/Dropbox/business/office/workspace/software/repositories/organizations/restudToolbox/package/respy/tests')
from codes.auxiliary import write_draws


num_periods = 2
max_draws = 1000
write_draws(num_periods, max_draws)

respy_obj = RespyCls('model.respy.ini')
respy_fort = solve(respy_obj)

respy_obj = RespyCls('model.respy.ini')
respy_obj.attr['version'] = 'PYTHON'
respy_pyth =  solve(respy_obj)

print(respy_pyth == respy_fort)
print(respy_pyth.check_equal_solution(respy_fort))






