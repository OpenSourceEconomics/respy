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

# Run workflowR
robupy_obj = read('model.robupy.ini')

robupy_obj = solve(robupy_obj)

import shlex











print('done')