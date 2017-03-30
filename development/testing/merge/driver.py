#!/usr/bin/env python
""" This script runs a series of tests that are required before merging a
candidate branch into master.
"""

import os
import respy
from respy.python.shared.shared_constants import ROOT_DIR
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = ROOT_DIR.replace('respy', '')
TEST_DIR = PACKAGE_DIR + '/development/testing/'

test_spec = dict()

test_spec['PYTEST'] = dict()
test_spec['PYTEST']['run'] = False

test_spec['REGRESSION'] = dict()
test_spec['REGRESSION']['run'] = False
test_spec['REGRESSION']['request'] = ('check', 1)
test_spec['REGRESSION']['is_background'] = True
test_spec['REGRESSION']['is_compile'] = False

test_spec['PROPERTY'] = dict()
test_spec['PROPERTY']['run'] = True
test_spec['PROPERTY']['request'] = ('run', 0.00001)
test_spec['PROPERTY']['is_background'] = True
test_spec['PROPERTY']['is_compile'] = False


for dirname in ['regression', 'property']:
    sys.path.insert(0, TEST_DIR + '/' + dirname)

from run_regression import run as run_regression
from run_property import run as run_property

# TODO: NOtifications ?
# TODO: Add runs in Python 2 and 3
# We can run the PYTEST battery.
if test_spec['PYTEST']['run']:
    respy.test()


if test_spec['REGRESSION']['run']:

    is_background = test_spec['REGRESSION']['is_background']
    is_compile = test_spec['REGRESSION']['is_compile']
    request = test_spec['REGRESSION']['request']

    is_success_regression = run_regression(request, is_compile, is_background)

if test_spec['PROPERTY']['run']:

    is_background = test_spec['PROPERTY']['is_background']
    is_compile = test_spec['PROPERTY']['is_compile']
    request = test_spec['PROPERTY']['request']

    run_property(request, is_compile, is_background)