#!/usr/bin/env python
"""This script runs a series of tests that are required for any pull request to be merged."""
import respy
import sys
import os
from socket import gethostname
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
for dirname in ["regression", "property", "release", "robustness", "parallelism"]:
    sys.path.insert(0, CURRENT_DIR + "/" + dirname)

from run_regression import run as run_regression
from run_property import run as run_property
from run_robustness import run as run_robustness
from run_parallelism import run as run_parallelism

# Here we specify the group of tests to run. Later we also pin down the details.
request_dict = dict()
request_dict["REGRESSION"] = True
request_dict["PROPERTY"] = True
request_dict["PYTEST"] = True
request_dict["ROBUSTNESS"] = True
request_dict["PARALLELISM"] = True

# determine whether to do a long or short run
short_run = gethostname() in ["socrates"]

# We need to specify the arguments for each of the tests.
test_spec = dict()
test_spec["PYTEST"] = dict()

test_spec["REGRESSION"] = dict()
if short_run is True:
    test_spec["REGRESSION"]["request"] = ("check", 50)
else:
    test_spec["REGRESSION"]["request"] = ("check", 10000)
test_spec["REGRESSION"]["is_background"] = False
test_spec["REGRESSION"]["is_compile"] = False
test_spec["REGRESSION"]["is_strict"] = True
test_spec["REGRESSION"]["num_procs"] = 3

test_spec["PROPERTY"] = dict()
if short_run is True:
    test_spec["PROPERTY"]["request"] = ("run", 0.05)
else:
    test_spec["PROPERTY"]["request"] = ("run", 12)
test_spec["PROPERTY"]["is_background"] = False
test_spec["PROPERTY"]["is_compile"] = False

test_spec["ROBUSTNESS"] = dict()
if short_run is True:
    test_spec["ROBUSTNESS"]["request"] = ("run", 0.05)
else:
    test_spec["ROBUSTNESS"]["request"] = ("run", 12)
test_spec["ROBUSTNESS"]["is_compile"] = False
test_spec["ROBUSTNESS"]["is_background"] = False
test_spec["ROBUSTNESS"]["keep_dataset"] = False
test_spec["ROBUSTNESS"]["num_procs"] = 3

test_spec["PARALLELISM"] = dict()
if short_run is True:
    test_spec["PARALLELISM"]["hours"] = 0.05
else:
    test_spec["PARALLELISM"]["hours"] = 12


if request_dict["PYTEST"]:
    print('\n\n', 'starting pytest', '\n\n')
    respy.test()
    print('\n\n', 'stopping pytest', '\n\n')

if request_dict["REGRESSION"]:
    print('\n\n', 'starting regression test', '\n\n')
    os.chdir("regression")
    run_regression(**test_spec["REGRESSION"])
    os.chdir(CURRENT_DIR)
    print('\n\n', 'stopping regression test', '\n\n')

if request_dict["PROPERTY"]:
    print('\n\n', 'starting property test', '\n\n')
    os.chdir("property")
    run_property(**test_spec["PROPERTY"])
    os.chdir(CURRENT_DIR)
    print('\n\n', 'stopping property test', '\n\n')

if request_dict["ROBUSTNESS"]:
    print('\n\n', 'starting robustness test', '\n\n')
    os.chdir("robustness")
    run_robustness(**test_spec["ROBUSTNESS"])
    os.chdir(CURRENT_DIR)
    print('\n\n', 'stopping robustness test', '\n\n')

if request_dict["PARALLELISM"]:
    print('\n\n', 'starting parallelism test', '\n\n')
    os.chdir("parallelism")
    run_parallelism(**test_spec["PARALLELISM"])
    os.chdir(CURRENT_DIR)
    print('\n\n', 'stopping parallelism test', '\n\n')
