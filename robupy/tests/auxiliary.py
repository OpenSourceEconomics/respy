""" Auxiliary functions for development test suite.
"""

# standard library
import logging
import socket
import shutil
import shlex
import glob
import os

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
    ''' Check input for estimation script.
    '''
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    hours = args.hours
    notification = args.notification

    # Assertions.
    assert (notification in [True, False])
    assert (isinstance(hours, float))
    assert (hours > 0.0)

    # Validity checks
    if notification:
        # Check that the credentials file is stored in the user's HOME directory.
        assert (os.path.exists(os.environ['HOME'] + '/.credentials'))

    # Finishing.
    return hours, notification

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

    cmd = cmd + ' > /dev/null 2>&1'

    os.system(cmd)

    os.chdir(tests_dir)

def check_ambiguity_optimization():
    """ This function checks that less than 5% of all optimization for each
    period fail.
    """
    def _process_cases(list_):
        """ Process cases and determine whether keyword or empty line.
        """
        # Antibugging
        assert (isinstance(list_, list))

        # Get information
        is_empty = (len(list_) == 0)

        if not is_empty:
            is_summary = (list_[0] == 'SUMMARY')
        else:
            is_summary = False

        # Antibugging
        assert (is_summary in [True, False])
        assert (is_empty in [True, False])

        # Finishing
        return is_empty, is_summary

    is_relevant = False

    # Check relevance
    if not os.path.exists('ambiguity.robupy.log'):
        return

    for line in open('ambiguity.robupy.log').readlines():

        # Split line
        list_ = shlex.split(line)

        # Determine special cases
        is_empty, is_summary = _process_cases(list_)

        # Applicability
        if is_empty:
            continue

        # Prepare dictionary
        if is_summary:
            is_relevant = True
            continue

        if not is_relevant:
            continue

        if list_[0] == 'Period':
            continue

        period, total, success, failure = list_

        total = success + failure

        if float(failure)/float(total) > 0.05:
            raise AssertionError



