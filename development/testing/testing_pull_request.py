#!/usr/bin/env python
"""This script runs a series of tests that are required for any pull request to be merged."""
import respy
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
for dirname in ['regression', 'property', 'release']:
    sys.path.insert(0, CURRENT_DIR + '/' + dirname)

from run_regression import run as run_regression
from run_property import run as run_property
from run_robustness import run as run_robustness

# Here we specify the group of tests to run. Later we also pin down the details.
request_dict = dict()
request_dict['REGRESSION'] = True
request_dict['PROPERTY'] = True
request_dict['PYTEST'] = True
request_dict['ROBUSTNESS'] = True

# We need to specify the arguments for each of the tests.
test_spec = dict()
test_spec['PYTEST'] = dict()

test_spec['REGRESSION'] = dict()
test_spec['REGRESSION']['request'] = ('check', 10000)
test_spec['REGRESSION']['is_background'] = False
test_spec['REGRESSION']['is_compile'] = False
test_spec['REGRESSION']['is_strict'] = True
test_spec['REGRESSION']['num_procs'] = 10

test_spec['PROPERTY'] = dict()
test_spec['PROPERTY']['request'] = ('run', 12)
test_spec['PROPERTY']['is_background'] = False
test_spec['PROPERTY']['is_compile'] = False

test_spec['ROBUSTNESS']['request'] = ('run', 12)
test_spec['ROBUSTNESS']['is_compile'] = False
test_spec['ROBUSTNESS']['is_background'] = False
test_spec['ROBUSTNESS']['num_procs'] = 1
test_spec['ROBUSTNESS']['keep_dataset'] = False

if request_dict['PYTEST']:
    respy.test()

if request_dict['REGRESSION']:
    os.chdir('regression')
    is_success_regression = run_regression(**test_spec['REGRESSION'])
    os.chdir(CURRENT_DIR)

if request_dict['PROPERTY']:
    os.chdir('property')
    run_property(**test_spec['PROPERTY'])
    os.chdir(CURRENT_DIR)

if request_dict['ROBUSTNESS']:
    os.chdir('robustness')
    run_property(**test_spec['ROBUSTNESS'])
    os.chdir(CURRENT_DIR)
