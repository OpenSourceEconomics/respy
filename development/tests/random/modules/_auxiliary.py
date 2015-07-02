""" Auxiliary functions for development test suite.
"""

# standard library
import logging
import socket
import shutil
import json
import glob
import os

# project library
from modules.clsMail import mailCls

''' Auxiliary functions.
'''


def start_logging():
    """ Start logging of performance.
    """

    # Initialize logger
    logger = logging.getLogger('DEV-TEST')
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler('logging.log', 'w')
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(' %(asctime)s     %(message)s \n', datefmt='%I:%M:%S %p')
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)

def distribute_input(parser):
    """ Check input for estimation script.
    """
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    hours = args.hours

    notification = args.notification

    # Assertions.
    assert (isinstance(hours, float))
    assert (hours > 0.0)

    assert (notification in [True, False])

    # Finishing.
    return hours, notification

def finish(dict_, HOURS, notification):
    """ Finishing up a run of the testing battery.
    """
    # Antibugging.
    assert (isinstance(dict_, dict))
    assert (notification in [True, False])

    # Auxiliary objects.
    hostname = socket.gethostname()

    with open('logging.log', 'a') as file_:

        file_.write(' Summary \n\n')

        str_ = '   Test {0:<10} Success {1:<10} Failures  {2:<10}\n'

        for label in sorted(dict_.keys()):

            success = dict_[label]['success']

            failure = dict_[label]['failure']

            file_.write(str_.format(label, success, failure))

        file_.write('\n')

    if notification:

        subject = ' ROBUPY: Completed Testing Battery '

        message = ' A ' + str(HOURS) +' hour run of the testing battery on @' + \
                  hostname + ' is completed.'

        mailObj = mailCls()

        mailObj.setAttr('subject', subject)

        mailObj.setAttr('message', message)

        mailObj.setAttr('attachment', 'logging.log')

        mailObj.lock()

        mailObj.send()

def cleanup():
    """ Cleanup after test battery.
    '"""
    files = []

    files = files + glob.glob('*.robupy.*')

    files = files + glob.glob('*.ini')

    files = files + glob.glob('*.pkl')

    files = files + glob.glob('*.txt')

    files = files + glob.glob('*.dat')

    for file_ in files:

        try:

            os.remove(file_)

        except OSError:

            pass

        try:

            shutil.rmtree(file_)

        except OSError:

            pass
