#!/usr/bin/env python
""" This module is used for profiling purposes.
"""

# standard library
import sys
import os

# ROBUPY
sys.path.insert(0, os.environ['ROBUPY'])
from robupy.solve import solve
from robupy.read import read


robupy_obj = read('model.robupy.ini')

solve(robupy_obj)
