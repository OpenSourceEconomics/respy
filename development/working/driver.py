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
estimate(respy_obj)


