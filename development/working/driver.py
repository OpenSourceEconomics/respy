#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
import numpy as np

import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from robupy import *

from robupy.python.py.auxiliary import opt_get_model_parameters
from robupy.python.py.auxiliary import opt_get_optim_parameters
from robupy.auxiliary import distribute_model_paras

# Read in baseline initialization file
#compile_package('--fortran --debug', False)

np.random.seed(123)

# Generate random initialization file
robupy_obj = read('original.robupy.ini')

solve(robupy_obj)

simulate(robupy_obj)

data_frame = process('data.robupy.dat', robupy_obj)

is_debug = robupy_obj.get_attr('is_debug')

model_paras = robupy_obj.get_attr('model_paras')

eps_cholesky = model_paras['eps_cholesky']

model_paras = robupy_obj.get_attr("model_paras")

args = distribute_model_paras(model_paras, is_debug)

x = opt_get_optim_parameters(*args, is_debug=is_debug)


# Construct auxiliary objects
args = opt_get_model_parameters(x, is_debug)

base = evaluate(robupy_obj, data_frame)

np.testing.assert_almost_equal(base, 0.18991613969942756)