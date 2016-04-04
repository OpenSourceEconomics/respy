#!/usr/bin/env python
""" I will now try to run some estimations.
"""


# standard library
import os
import sys

# ROOT DIRECTORY
sys.path.insert(0, os.environ['ROBUPY'])
# ROOT DIRECTORY
ROOT_DIR = os.environ['ROBUPY']
ROOT_DIR = ROOT_DIR + '/robupy/tests'

sys.path.insert(0, ROOT_DIR)

# project library
from robupy.python.evaluate.evaluate_python import pyth_evaluate
from robupy.fortran.f2py_library import f2py_evaluate

from robupy.fortran.fortran import fort_evaluate
from robupy.tests.codes.auxiliary import write_draws

from robupy.python.evaluate.evaluate_auxiliary import check_input
from robupy.python.evaluate.evaluate_auxiliary import check_output

from robupy.python.shared.shared_auxiliary import distribute_class_attributes
from robupy.python.shared.shared_auxiliary import distribute_model_paras
from robupy.python.shared.shared_auxiliary import create_draws

from robupy import simulate, read, solve, evaluate
robupy_obj = read('model.robupy.ini')

print('starting to simulate')
data_frame = simulate(robupy_obj)

num_periods = robupy_obj.get_attr('num_periods')
write_draws(num_periods, 1000)

for version in ['PYTHON', 'FORTRAN']:
    print('\n', version, '\n')
    robupy_obj.unlock()

    robupy_obj.set_attr('version', version)

    robupy_obj.lock()

    evaluate(robupy_obj, data_frame)