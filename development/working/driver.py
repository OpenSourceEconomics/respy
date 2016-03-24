#!/usr/bin/env python
""" I will now try to run some estimations.
"""


# standard library
import numpy as np
from scipy.optimize import minimize
import sys
import os

# ROOT DIRECTORY
sys.path.insert(0, os.environ['ROBUPY'])
# ROOT DIRECTORY
ROOT_DIR = os.environ['ROBUPY']
ROOT_DIR = ROOT_DIR + '/robupy/tests'

sys.path.insert(0, ROOT_DIR)

# testing codes
from codes.auxiliary import cleanup_robupy_package
from codes.auxiliary import build_testing_library
from codes.auxiliary import build_robupy_package

from robupy import simulate, read, solve, process, evaluate, estimate
#build_robupy_package(False)

robupy_obj = read('test.robupy.ini')

# First, I simulate a dataset.
robupy_obj = solve(robupy_obj)

val = robupy_obj.get_attr('periods_emax')[0, 0]
#np.testing.assert_allclose(3.664605209230335, val)

simulate(robupy_obj)
val, _ = evaluate(robupy_obj, process(robupy_obj))
#np.testing.assert_allclose(8.73671639678513, val)

data_frame = process(robupy_obj)

#estimate(robupy_obj, data_frame)