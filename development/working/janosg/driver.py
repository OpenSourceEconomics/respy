#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('./waf distclean; ./waf configure build --debug') == 0
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
from respy._scripts.scripts_compare import scripts_compare
from codes.auxiliary import simulate_observed
from codes.auxiliary import write_draws

from codes.random_init import generate_init
from respy.python.shared.shared_auxiliary import dist_class_attributes, \
    get_conditional_probabilities, back_out_systematic_wages
from respy.python.process.process_python import process
#write_draws(5, 5000)

from codes.auxiliary import write_types
from codes.auxiliary import write_edu_start
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR

sys.path.insert(0, TEST_RESOURCES_DIR)
#import f2py_interface as fort_debug

np.random.seed(123)

respy_obj = RespyCls("model.respy.ini")
#simulate(respy_obj)

estimate(respy_obj)
