#!/usr/bin/env python
""" I will now try to run some estimations.
"""


# ROOT DIRECTORY
# standard library
import os
import sys

# ROOT DIRECTORY
sys.path.insert(0, os.environ['ROBUPY'])
sys.path.insert(0, os.environ['STRUCT_RECOMPUTATION'])
from robupy.python.estimate.estimate_auxiliary import dist_optim_paras

# project library
from robupy.python.evaluate.evaluate_python import pyth_evaluate
from robupy.fortran.f2py_library import f2py_evaluate

from robupy.fortran.fortran import fort_evaluate
from robupy.tests.codes.auxiliary import write_draws

from robupy.python.evaluate.evaluate_auxiliary import check_input
from robupy.python.evaluate.evaluate_auxiliary import check_output

from robupy.python.shared.shared_auxiliary import dist_class_attributes
from robupy.python.shared.shared_auxiliary import dist_model_paras
from robupy.python.shared.shared_auxiliary import create_draws

from robupy import simulate, read, solve, evaluate, estimate
robupy_obj = read('model.robupy.ini')
