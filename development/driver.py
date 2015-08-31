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
HOME = os.getcwd()


for _ in range(1):

    # Re-compile ROBUPY package
    os.chdir('/home/peisenha/robustToolbox/package/robupy')

    os.system('./waf distclean')

    os.system('./waf configure build --fast')

    os.chdir(HOME)

    # Run
    os.system('robupy-solve')