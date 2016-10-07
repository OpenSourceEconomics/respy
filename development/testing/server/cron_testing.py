#!/usr/bin/env python
"""  This script can be registered as a cron job on the development server.
The idea is to have this running as a routine test battery.
"""

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import subprocess
import smtplib
import socket
import json
import sys
import os

from auxiliary_shared import compile_package
from respy.python.shared.shared_constants import ROOT_DIR

# We are using features for the automatic creation of the virtual environment
# for the release testing which are only available in Python 3.
from config import python3_exec

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = ROOT_DIR.replace('respy', '')

# This is the location to specify the details of the testing request.
NUM_REGRESSION_TESTS = 500
HRS_PROPERTY_TESTS = 6
HRS_RELEASE_TESTS = 6
NOTIFICATION = True


def finalize_message(ret, msg):
    """ This function simply finalizes the message for the notification
    depending on whether the requested tests were run successfully or not.
    """
    if ret == 0:
        msg += ' successfully.'
    else:
        msg += ' unsuccessfully.'

    return msg


def send_notification(msg_regression, msg_release, msg_property):
    """ Send notification about testing results.
    """
    subject = ' RESPY: Testing Server '

    message = ' A routine test battery just completed on the dedicated RESPY ' \
        + 'server. Here are the results:\n\n 1) ' + msg_regression + \
        '\n\n 2) ' + msg_release + '\n\n 3) ' + msg_property + \
        '\n\n Happy Testing, The respy Team'

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

    f = open('property.respy.info', 'r')

    os.chdir(CURRENT_DIR)

    attached = MIMEText(f.read())

    attached.add_header('Content-Disposition', 'attachment', filename='property.respy.info')

    msg.attach(attached)

    # Message
    message = MIMEText(message, 'plain')

    msg.attach(message)

    # Send
    server.sendmail(hostname, 'eisenhauer@policy-lab.org', msg.as_string())

    # Disconnect
    server.quit()


# Fresh setup
compile_package()

###############################################################################
# Regression Testing
###############################################################################
os.chdir(PACKAGE_DIR + '/development/testing/regression')
cmd = [python3_exec, 'driver.py', '--request', 'check', str(NUM_REGRESSION_TESTS)]
ret = subprocess.call(cmd + ['--background'])
os.chdir(CURRENT_DIR)

msg_regression = finalize_message(ret, 'We ran the requested regression tests')

###############################################################################
# Release Testing
###############################################################################
os.chdir(PACKAGE_DIR + '/development/testing/releases')
cmd = [python3_exec, 'driver.py', '--request', 'run', str(HRS_RELEASE_TESTS), '--create']
ret = subprocess.call(cmd + ['--background'])
os.chdir(CURRENT_DIR)

msg_release = finalize_message(ret, 'We ran the requested release tests')
###############################################################################
# Property-based Testing
###############################################################################
os.chdir(PACKAGE_DIR + '/development/testing/property')
cmd = [python3_exec, 'driver.py', '--request', 'run', str(HRS_PROPERTY_TESTS)]
ret = subprocess.call(cmd + ['--background'])
os.chdir(CURRENT_DIR)

msg_property = 'We ran the requested property-based tests. ' \
               'The results are attached.'

###############################################################################
# Send report
###############################################################################
if NOTIFICATION and os.path.exists(os.environ['HOME'] + '/.credentials'):
    send_notification(msg_regression, msg_release, msg_property)
