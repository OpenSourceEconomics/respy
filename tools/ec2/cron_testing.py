#!/usr/bin/env python
"""  This script can be registered as a cron job on the development server.
The idea is to have the test battery executed automatically if the CPU load
allows.
"""

# standard library
import os

# Specify request
HOURS, NOTIFICATION = 0.0000001, True

# Get some basic information about the system.
LOADAVG = os.getloadavg()[2]

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('/tools/ec2', '')

# Move into the testing directory
os.chdir(PACKAGE_DIR + '/development/testing/automated')

# Execute script on the development server.
is_available = LOADAVG < 0.5
VIRUTALENV_PYTHON = '$HOME/.envs/restudToolbox3/bin/python'

if is_available:
    cmd = VIRUTALENV_PYTHON
    cmd += ' run.py --compile --hours ' + str(HOURS)
    if NOTIFICATION:
        cmd += '  --notification'

    os.system(cmd)





