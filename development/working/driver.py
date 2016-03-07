#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
import numpy as np

import scipy
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# RobuPy library
from robupy import *

from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import create_disturbances
from robupy.auxiliary import distribute_model_paras

from robupy.python.solve_python import solve_python_bare
from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

# testing battery
from modules.auxiliary import compile_package

compile_package('--fortran --debug', False)

# Perform toolbox actions
robupy_obj = read('test.robupy.ini')

robupy_obj = solve(robupy_obj)


periods_emax = robupy_obj.get_attr('periods_emax')


np.testing.assert_equal(periods_emax[0, 0], 3.5618948509283999)