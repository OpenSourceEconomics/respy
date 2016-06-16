#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
if True:
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
import numpy as np
import pickle as pkl


respy_obj = RespyCls('model.respy.ini')
simulate(respy_obj)
estimate(respy_obj)
