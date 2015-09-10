""" Auxiliary functions for development test suite.
"""

# standard library
import numpy as np

import logging
import socket
import shutil
import shlex
import glob
import os

# project library
from modules.clsMail import MailCls

''' Auxiliary functions.
'''


def write_disturbances(init_dict):
    """ Write out disturbances to potentially align the different
    implementations of the model
    """

    # Let us write out the disturbances to a file so that they can be aligned
    # between the alternative implementations
    num_draws = init_dict['SOLUTION']['draws']
    num_periods = init_dict['BASICS']['periods']
    shocks = init_dict['SHOCKS']

    periods_eps_relevant = np.random.multivariate_normal(np.zeros(4),
                            shocks, (num_periods, num_draws))

    for period in range(num_periods):
        for j in [0, 1]:
            periods_eps_relevant[period, :, j] = \
                np.exp(periods_eps_relevant[period, :, j])

    with open('disturbances.txt', 'w') as file_:
        for period in range(num_periods):
            for i in range(num_draws):
                line = ' {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}\n'.format(
                    *periods_eps_relevant[period, i, :])
                file_.write(line)


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
    assert (notification in [True, False])
    assert (isinstance(hours, float))
    assert (hours > 0.0)

    # Validity checks
    if notification:
        # Check that the credentials file is stored in the user's HOME directory.
        assert (os.path.exists(os.environ['HOME'] + '/.credentials'))

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

        mail_obj = MailCls()

        mail_obj.set_attr('subject', subject)

        mail_obj.set_attr('message', message)

        mail_obj.set_attr('attachment', 'logging.log')

        mail_obj.lock()

        mail_obj.send()

def cleanup():
    """ Cleanup after test battery.
    '"""
    files = []

    files = files + glob.glob('*.robupy.*')

    files = files + glob.glob('*.ini')

    files = files + glob.glob('*.pkl')

    files = files + glob.glob('*.txt')

    files = files + glob.glob('*.dat')

    files = files + glob.glob('modules/dp3asim')

    files = files + glob.glob('.write_out')

    for file_ in files:

        try:

            os.remove(file_)

        except OSError:

            pass

        try:

            shutil.rmtree(file_)

        except OSError:

            pass

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

    cmd = './waf configure build --debug'

    if which == 'fast':
        cmd += ' --fast'

    cmd += ' > /dev/null 2>&1'

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


def transform_robupy_to_restud(init_dict):
    """ Transform a ROBUPY initialization file to a RESTUD file.
    """
    # Ensure restrictions
    assert (init_dict['AMBIGUITY']['level'] == 0.00)
    assert (init_dict['EDUCATION']['start'] == 10)
    assert (init_dict['EDUCATION']['max'] == 20)

    with open('in.txt', 'w') as file_:

        # Write out some basic information about the problem.
        num_agents = init_dict['SIMULATION']['agents']
        num_periods = init_dict['BASICS']['periods']
        num_draws = init_dict['SOLUTION']['draws']
        file_.write(' {0:03d} {1:05d} {2:06d} {3:06f}'
            ' {4:06f}\n'.format(num_periods, num_agents, num_draws,-99.0,
            500.0))

        # Write out coefficients for the two occupations.
        for label in ['A', 'B']:
            coeffs = [init_dict[label]['int']] + init_dict[label]['coeff']
            line = ' {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f}  {4:10.6f}' \
                    ' {5:10.6f}\n'.format(*coeffs)
            file_.write(line)

        # Write out coefficients for education and home payoffs as well as
        # the discount factor. The intercept is scaled. This is later undone
        # again in the original FORTRAN code.
        edu_int = init_dict['EDUCATION']['int'] / 1000; edu_coeffs = [edu_int]
        home = init_dict['HOME']['int'] / 1000
        for j in range(2):
            edu_coeffs += [-init_dict['EDUCATION']['coeff'][j] / 1000]
        delta = init_dict['BASICS']['delta']
        coeffs = edu_coeffs + [home, delta]
        line = ' {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f}' \
                ' {4:10.6f}\n'.format(*coeffs)
        file_.write(line)

        # Write out coefficients of correlation, which need to be constructed
        # based on the covariance matrix.
        shocks = init_dict['SHOCKS']; rho = np.identity(4)
        rho_10 = shocks[1][0] / (np.sqrt(shocks[1][1]) * np.sqrt(shocks[0][0]))
        rho_20 = shocks[2][0] / (np.sqrt(shocks[2][2]) * np.sqrt(shocks[0][0]))
        rho_30 = shocks[3][0] / (np.sqrt(shocks[3][3]) * np.sqrt(shocks[0][0]))
        rho_21 = shocks[2][1] / (np.sqrt(shocks[2][2]) * np.sqrt(shocks[1][1]))
        rho_31 = shocks[3][1] / (np.sqrt(shocks[3][3]) * np.sqrt(shocks[1][1]))
        rho_32 = shocks[3][2] / (np.sqrt(shocks[3][3]) * np.sqrt(shocks[2][2]))
        rho[1, 0] = rho_10; rho[2, 0] = rho_20; rho[3, 0] = rho_30
        rho[2, 1] = rho_21; rho[3, 1] = rho_31; rho[3, 2] = rho_32
        for j in range(4):
            line = ' {0:10.5f} {1:10.5f} {2:10.5f} ' \
                   ' {3:10.5f}\n'.format(*rho[j, :])
            file_.write(line)

        # Write out standard deviations. Scaling for standard deviations in the
        # education and home equation required. This is undone again in the
        # original FORTRAN code.
        sigmas = np.sqrt(np.diag(shocks)); sigmas[2:] = sigmas[2:]/1000
        line = '{0:10.5f} {1:10.5f} {2:10.5f} {3:10.5f}\n'.format(*sigmas)
        file_.write(line)