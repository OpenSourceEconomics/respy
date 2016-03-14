""" Auxiliary functions for development test suite.
"""

# standard library
import numpy as np

import logging
import socket
import shutil
import shlex
import glob
import sys
import os

# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy import read

from robupy.python.solve_python import _create_state_space

# project library
from material.clsMail import MailCls

''' Auxiliary functions.
'''

def distribute_model_description(robupy_obj, *args):
    """ This function distributes the model description.
    """

    ret = []

    for arg in args:
        ret.append(robupy_obj.get_attr(arg))

    return ret

def write_interpolation_grid(file_name):
    """ Write out an interpolation grid that can be used across
    implementations.
    """
    # Process relevant initialization file
    robupy_obj = read(file_name)

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    num_points = robupy_obj.get_attr('num_points')

    edu_start = robupy_obj.get_attr('edu_start')

    is_python = robupy_obj.get_attr('is_python')

    edu_max = robupy_obj.get_attr('edu_max')

    min_idx = robupy_obj.get_attr('min_idx')

    # Determine maximum number of states
    _, states_number_period, _, max_states_period = \
        _create_state_space(num_periods, edu_start, edu_max, min_idx, is_python)

    # Initialize container
    booleans = np.tile(True, (max_states_period, num_periods))

    # Iterate over all periods
    for period in range(num_periods):

        # Construct auxiliary objects
        num_states = states_number_period[period]
        any_interpolation = (num_states - num_points) > 0

        # Check applicability
        if not any_interpolation:
            continue

        # Draw points for interpolation
        indicators = np.random.choice(range(num_states),
                            size=num_states - num_points, replace=False)

        # Replace indicators
        for i in range(num_states):
            if i in indicators:
                booleans[i, period] = False

    # Write out to file
    np.savetxt('interpolation.txt', booleans, fmt='%s')

    # Some information that is useful elsewhere.
    return max_states_period



def write_disturbances(num_periods, max_draws):
    """ Write out disturbances to potentially align the different
    implementations of the model. Note that num draws has to be less or equal
    to the largest number of requested random deviates.
    """
    # Draw standard deviates
    standard_deviates = np.random.multivariate_normal(np.zeros(4),
        np.identity(4), (num_periods, max_draws))

    # Write to file to they can be read in by the different implementations.
    with open('disturbances.txt', 'w') as file_:
        for period in range(num_periods):
            for i in range(max_draws):
                line = ' {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}\n'.format(
                    *standard_deviates[period, i, :])
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

    ''' Clean main.
    '''
    files += glob.glob('.waf*')

    files += glob.glob('.pkl*')

    files += glob.glob('.txt*')

    files += glob.glob('.grm.*')

    files += glob.glob('*.pkl')

    files += glob.glob('*.txt')

    files += glob.glob('*.out')

    files += glob.glob('test*.ini')

    files += glob.glob('*.pyc')

    files += glob.glob('.seed')

    files += glob.glob('.dat*')

    files += glob.glob('*.robupy.*')

    files += glob.glob('*.robufort.*')

    ''' Clean modules.
        '''
    files += glob.glob('modules/*.out*')

    files += glob.glob('modules/*.pyc')

    files += glob.glob('modules/*.mod')

    files += glob.glob('modules/dp3asim')

    files += glob.glob('modules/lib')

    files += glob.glob('modules/include')

    files += glob.glob('modules/*.so')

    files += glob.glob('*.dat')

    files += glob.glob('.restud.testing.scratch')

    files += glob.glob('modules/*.o')

    files += glob.glob('modules/__pycache__')

    for file_ in files:

        try:

            os.remove(file_)

        except:

            try:

                shutil.rmtree(file_)

            except:

                pass


def compile_package(options, is_hidden):
    """ Compile toolbox
    """
    # Auxiliary objects
    package_dir = os.environ['ROBUPY'] + '/robupy'
    tests_dir = os.getcwd()

    # Compile package
    os.chdir(package_dir)

    for i in range(1):

        os.system('./waf distclean > /dev/null 2>&1')

        cmd = './waf configure build '

        cmd += options

        if is_hidden:
            cmd += ' > /dev/null 2>&1'

        os.system(cmd)

        # In a small number of cases the build process seems to fail for no
        # reason.
        try:
            import robupy.python.f2py.f2py_library
            break
        except ImportError:
            pass

        if i == 10:
            raise AssertionError

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

def build_f2py_testing(is_hidden):
    """ Build the F2PY testing interface for testing.f
    """
    os.chdir('material')

    # Build static library
    compiler_options = '-O3 -fpic'

    files = ['robufort_constants.f90', 'robufort_auxiliary.f90',
             'robufort_slsqp.f', 'robufort_emax.f90', 'robufort_risk.f90',
             'robufort_ambiguity.f90', 'robufort_library.f90',
             'robufort_testing.f90']

    for file_ in files:

        cmd = 'gfortran ' + compiler_options + ' -c ' + file_

        if is_hidden:
            cmd += ' > /dev/null 2>&1'
        os.system(cmd)

    os.system('ar crs libfort_testing.a *.o *.mod')

    # Prepare directory structure
    for dir_ in ['include', 'lib']:
        try:
            shutil.rmtree(dir_)
        except OSError:
                pass
        try:
            os.makedirs(dir_)
        except OSError:
            pass

    # Finalize static library
    module_files = glob.glob('*.mod')
    for file_ in module_files:
        shutil.move(file_, 'include/')
    shutil.move('libfort_testing.a', 'lib/')

    # Build interface
    cmd = 'f2py3 -c -m  f2py_testing f2py_interface_testing.f90 -Iinclude ' \
          ' -Llib -lfort_testing -L/usr/lib/lapack -llapack'

    if is_hidden:
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    # Finish
    os.chdir('../')

