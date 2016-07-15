#!/usr/bin/env python
"""  This script can be registered as a cron job on the development server.
The idea is to have this running as a routine test battery.
"""

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np

import subprocess
import smtplib
import socket
import json
import sys
import os

# Get some basic information about the system and only start the work if
# server not in other use.
if socket.gethostname() != 'pontos':
    LOADAVG = os.getloadavg()[2]
    is_available = LOADAVG < 0.5
    if not is_available:
        sys.exit()

# Specify request
HOURS, NOTIFICATION = 6, True

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = CURRENT_DIR.replace('/tools/ec2', '')

sys.path.insert(0, PACKAGE_DIR + '/development/testing/_modules')
from config import python2_exec
from config import python3_exec

###############################################################################
# Run the PYTHON 2 test vault.s
###############################################################################
os.chdir(PACKAGE_DIR + '/development/testing/regression')
test_vault_2_ret = subprocess.call([python2_exec, 'run.py', '--version', '2', '--request', 'check', '--background'])

test_vault_2_msg = '1) The PYTHON 2 testing vault was run '
if test_vault_2_ret == 0:
    test_vault_2_msg += 'successfully.'
else:
    test_vault_2_msg += 'unsuccessfully.'

os.chdir(CURRENT_DIR)
###############################################################################
# Run the PYTHON 3 test vault.
###############################################################################
os.chdir(PACKAGE_DIR + '/development/testing/regression')
test_vault_3_ret = subprocess.call([python2_exec, 'run.py', '--version', '3', '--request', 'check', '--background'])

test_vault_3_msg = '2) The PYTHON 3 testing vault was run '
if test_vault_3_ret == 0:
    test_vault_3_msg += 'successfully.'
else:
    test_vault_3_msg += 'unsuccessfully.'

os.chdir(CURRENT_DIR)
###############################################################################
# Run the regular test battery
###############################################################################
# Move into the testing directory
os.chdir(PACKAGE_DIR + '/development/testing/property')

# Execute script on the development server.
cmd = np.random.choice([python3_exec, python2_exec])
cmd += ' run.py --hours ' + str(HOURS) + ' --background'
subprocess.call(cmd, shell=True)

###############################################################################
# Send report
###############################################################################
if NOTIFICATION:

    subject = ' RESPY: Testing Server '

    message = ' A routine test battery just completed on the dedicated RESPY server. Here are the results:\n\n' \
              + test_vault_2_msg + '\n\n' + test_vault_3_msg + '\n\n' \
              '3) We also ran a ' + str(HOURS) + ' hour run of the testing battery. The results are attached.\n\n' \
              'Happy Testing, The respy Team'

    # Process credentials
    dict_ = json.load(open(os.environ['HOME'] + '/.credentials'))
    username = dict_['username']
    password = dict_['password']

    # Connect to GMAIL
    server = smtplib.SMTP('smtp.gmail.com:587')

    server.starttls()

    server.login(username, password)

    # Formatting
    msg = MIMEMultipart('alternative')

    msg['Subject'], msg['From'] = subject, socket.gethostname()

    hostname = socket.gethostname()

    # Attachment
    os.chdir(PACKAGE_DIR + '/development/testing/property')

    f = open('report.testing.log', 'r')

    os.chdir(CURRENT_DIR)

    attached = MIMEText(f.read())

    attached.add_header('Content-Disposition', 'attachment', filename='report.testing.log')

    msg.attach(attached)

    # Message
    message = MIMEText(message, 'plain')

    msg.attach(message)

    # Send
    server.sendmail(hostname, 'eisenhauer@policy-lab.org', msg.as_string())

    # Disconnect
    server.quit()


