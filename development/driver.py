#!/usr/bin/env python

""" This module is used for the development setup.
"""

# project library
import time
import sys
import os

sys.path.insert(0, os.environ['ROBUPY'])

# project library
from robupy import *

# Relative Criterion
robupy_obj = read('tests/fixed/third.robupy.ini')

robupy_obj = solve(robupy_obj)

simulate(robupy_obj)

cleanup()

# Assert unchanged value
assert abs(robupy_obj.get_attr('emax')[0, :1] - 55931.87437459) < 0.00000001
