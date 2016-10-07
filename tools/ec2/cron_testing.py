#!/usr/bin/env python
"""  This script can be registered as a cron job on the development server.
The idea is to have this running as a routine test battery.
"""


import subprocess
import socket
import sys
import os

from auxiliary_shared import compile_package

# We are using features for the automatic creation of the virtual environment
# for the release testing which are only available in Python 3.
if int(sys.version[0]) < 3:
    raise AssertionError('Please use Python 3')

# Get some basic information about the system and only start the work if
# server not in other use.
if socket.gethostname() != 'pontos':
    LOADAVG = os.getloadavg()[2]
    is_available = LOADAVG < 0.5
    if not is_available:
        sys.exit()

NUM_REGRESSION_TESTS = 1
HRS_RELEASE_TESTS = 0.001
HRS_PROPERTY_TESTS = 0.001

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = CURRENT_DIR.replace('/tools/ec2', '')

# Fresh setup
compile_package()

###############################################################################
# Regression Testing
###############################################################################
os.chdir(PACKAGE_DIR + '/development/testing/regression')
subprocess.call(['./driver.py', '--request', 'check', str(NUM_REGRESSION_TESTS)])
os.chdir(CURRENT_DIR)
###############################################################################
# Release Testing
###############################################################################
os.chdir(PACKAGE_DIR + '/development/testing/releases')
subprocess.call(['./driver.py', '--request', 'run', str(HRS_RELEASE_TESTS)])
os.chdir(CURRENT_DIR)
###############################################################################
# Property-based Testing
###############################################################################
os.chdir(PACKAGE_DIR + '/development/testing/property')
subprocess.call(['./driver.py', '--request', 'run', str(HRS_PROPERTY_TESTS)])
os.chdir(CURRENT_DIR)

