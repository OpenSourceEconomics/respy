#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
from scipy.stats import norm

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
from robupy.auxiliary import distribute_model_paras
from modules.auxiliary import write_interpolation_grid
from modules.auxiliary import write_disturbances

from robupy.python.solve_python import solve_python_bare
from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

from robupy.python.evaluate_python import _evaluate_python_bare, evaluate_criterion_function
# testing battery
from modules.auxiliary import compile_package


#print('not recompiling')
compile_package('--fortran --debug', False)

np.random.seed(321)

for _ in range(1000):
    constraints = dict()

    # THIS IS REQUIRED
    constraints['apply'] = False

    max_draws = np.random.random_integers(10, 100)
    constraints['max_draws'] = max_draws

    init_dict = generate_init(constraints)
    num_periods = init_dict['BASICS']['periods']

    write_disturbances(num_periods, max_draws)

    base = None

    for version in ['PYTHON', 'F2PY', 'FORTRAN']:


        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        data_frame = simulate(robupy_obj)

        # TODO: THIS CAN BE REWORKED?
        #robupy_obj.unlock()

        robupy_obj.attr['version'] = version
        robupy_obj.attr['is_python'] = version == 'PYTHON'
        #robupy_obj.lock()

        robupy_obj, eval = evaluate(robupy_obj, data_frame)

        if base is None:
            base = eval

        print('outside', version, eval)
        np.testing.assert_allclose(base, eval)

    try:
        os.unlink('disturbances.txt')
        os.unlink('interpolation.txt')
    except:
        pass

# 1.06544087763
# 1.3197840859728134