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


''' Auxiliary functions
'''

def compile_package(which):
    """ Compile toolbox
    """
    # Antibugging
    assert (which in ['fast', 'slow'])

    # Auxiliary objects
    package_dir = os.environ['ROBUPY'] + '/robupy'
    tests_dir = os.getcwd()

    # Compile package
    os.chdir(package_dir)

    os.system('./waf distclean > /dev/null 2>&1')

    cmd = './waf configure build'

    if which == 'fast':
        cmd += ' --fast'

    cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    os.chdir(tests_dir)

''' Main algorithm
'''
for which in ['slow', 'fast']:

    print('\n')

    compile_package(which)

    start_time = time.time()

    os.system('robupy-solve')

    end_time = time.time()

    print(' ' + which + ' ', (end_time - start_time)/60)

print('\n')
