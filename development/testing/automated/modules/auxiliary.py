
# standard library
from datetime import datetime

import numpy as np

import importlib
import fnmatch
import string
import socket
import random
import shutil
import glob
import sys
import os

# testing library
from modules.clsMail import MailCls

# RESPY directory. This allows to compile_ the debug version of the FORTRAN
# program.
RESPY_DIR = os.path.dirname(os.path.realpath(__file__))
RESPY_DIR = RESPY_DIR.replace('development/testing/automated/modules','') + 'respy'

# package imports
from respy.python.shared.shared_constants import IS_PARALLEL

''' Auxiliary functions
'''


def get_testdir(length):
    """ This function creates a random string that is used as the testing
    subdirectory.
    """
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def compile_package():
    """ Compile RESPY package in debug mode.
    """
    current_directory = os.getcwd()
    os.chdir(RESPY_DIR)
    os.system('./waf distclean; ./waf configure build --debug')
    os.chdir(current_directory)


def finalize_testing_record(full_test_record):
    """ Add the temporary file with the information about tracebacks to the
    main report.
    """

    # Count total and failed tests.
    total_tests, failed_tests = 0, 0
    for module in full_test_record.keys():
        for method in full_test_record[module].keys():
            total_tests += sum(full_test_record[module][method])
            failed_tests += full_test_record[module][method][1]

    # Indicate that test run is finished
    with open('report.testing.log', 'a') as log_file:
        log_file.write('\tRUN COMPLETED\n\n')
        fmt_ = '\t\t{0[0]:<15}{0[1]:>9}\n\n'
        log_file.write(fmt_.format(['TOTAL TESTS', total_tests]))
        log_file.write(fmt_.format(['FAILED TESTS', failed_tests]))

    # Aggregate information from temporary files.
    with open('report.testing.log', 'a') as outfile:
        outfile.write('\n---------------------------------------'
                      '-----------------------------------------\n\n')
        for line in open('tracebacks.testing.tmp'):
            outfile.write(line)
    os.unlink('tracebacks.testing.tmp')


def update_testing_record(module, method, seed_test, is_success, msg,
        full_test_record, start, timeout):
    """ Maintain a record about the outcomes of the testing efforts.
    """
    # Formatting of time objects.
    start_time = start.strftime("%Y-%m-%d %H:%M:%S")
    end_time = (start + timeout).strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write out overview information such as the number of successful and
    # failed test runs.
    with open('report.testing.log', 'w') as log_file:
        # Write out some header information.
        log_file.write('\n\n')
        string = '\t{0[0]:<15}{0[1]:<20}\n\n'
        log_file.write(string.format(['START', start_time]))
        log_file.write(string.format(['FINISH', end_time]))
        log_file.write(string.format(['UPDATE', current_time]))

        # Note the Python version used during execution.
        string = '\t{0[0]:<15}{0[1]}.{0[2]}.{0[3]}\n\n'
        log_file.write(string.format(['PYTHON'] + list(sys.version_info[:3])))

        log_file.write('\n\n')
        # Iterate over all modules. There is a potential conflict in the
        # namespace.
        for module_ in full_test_record.keys():
            string = '\t{0[0]:<29}{0[1]:<20}{0[2]:<20} \n\n'
            log_file.write(string.format([module_, 'Success', 'Failure']))
            # Iterate over all methods in the particular module.
            for method_ in sorted(full_test_record[module_]):
                string = '\t\t{0[0]:<25}{0[1]:<20}{0[2]:<20} \n'
                success, failure = full_test_record[module_][method_]
                log_file.write(string.format([method_, success, failure]))

            log_file.write('\n\n')

    # Special care for failures and tracebacks
    if not os.path.exists('tracebacks.testing.tmp'):
        open('tracebacks.testing.tmp', 'a').close()

    if not is_success:
        # Write out the traceback message to file for future inspection.
        with open('tracebacks.testing.tmp', 'a') as log_file:
            string = 'MODULE {0[0]:<25} METHOD {0[1]:<25} SEED: {0[' \
                     '2]:<10} \n\n'
            log_file.write(string.format([module, method, seed_test]))
            log_file.write(msg)
            log_file.write('\n---------------------------------------'
                           '-----------------------------------------\n\n')


def get_test_dict(test_dir):
    """ This function constructs a dictionary with the recognized test
    modules as the keys. The corresponding value is a list with all test
    methods inside that module.
    """
    # Process all candidate modules.
    current_directory = os.getcwd()
    os.chdir(test_dir)
    test_modules = []
    for test_file in glob.glob('test_*.py'):
        test_module = test_file.replace('.py', '')
        test_modules.append(test_module)
    os.chdir(current_directory)

    # Given the modules, get all tests methods.
    test_dict = dict()
    for test_module in test_modules:
        test_dict[test_module] = []
        mod = importlib.import_module(test_module.replace('.py', ''))
        candidate_methods = dir(mod.TestClass)
        for candidate_method in candidate_methods:
            if 'test_' in candidate_method:
                test_dict[test_module].append(candidate_method)

    # If the parallel version is not available, we remove the parallel tests.
    if not IS_PARALLEL:
        del test_dict['test_parallels']


    # Finishing
    return test_dict


def get_random_request(test_dict):
    """ This function extracts a random request from the dictionary.
    """
    # Create a list of tuples. Each tuple denotes a unique combination of
    # a module and a test contained in it.
    candidates = []
    for key_ in sorted(test_dict.keys()):
        for value in test_dict[key_]:
            candidates.append((key_, value))
    # Draw a random combination.
    index = np.random.randint(0, len(candidates))
    module, method = candidates[index]

    # Finishing
    return module, method


def distribute_input(parser):
    """ Check input for estimation script.
    """
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    notification = args.notification
    compile_ = args.compile
    hours = args.hours

    # Assertions.
    assert (notification in [True, False])
    assert (isinstance(hours, float))
    assert (compile_ in [True, False])
    assert (hours > 0.0)

    # Validity checks
    if notification:
        # Check that the credentials file is stored in the user's
        # HOME directory.
        assert (os.path.exists(os.environ['HOME'] + '/.credentials'))

    # Finishing.
    return hours, notification, compile_


def send_notification(hours):
    """ Finishing up a run of the testing battery.
    """
    # Auxiliary objects.
    hostname = socket.gethostname()

    subject = ' RESPY: Completed Testing Battery '

    message = ' A ' + str(hours) + ' hour run of the testing battery on @' + \
              hostname + ' is completed.'

    mail_obj = MailCls()

    mail_obj.set_attr('subject', subject)

    mail_obj.set_attr('message', message)

    mail_obj.set_attr('attachment', 'report.testing.log')

    mail_obj.lock()

    mail_obj.send()


def cleanup_testing_infrastructure(keep_results):
    """ This function cleans up before and after a testing run. If requested,
    the log file is retained.
    """
    # Collect all candidates files and directories.
    matches = []
    for root, dirnames, filenames in os.walk('.'):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '.*'):
            matches.append(os.path.join(root, filename))
        for dirname in fnmatch.filter(dirnames, '*'):
            matches.append(os.path.join(root, dirname))

    # Remove all files, unless explicitly to be saved.
    if keep_results:
        for fname in ['./report.testing.log', './tracebacks.testing.tmp']:
            if fname in matches:
                matches.remove(fname)

    # Iterate over all remaining files.
    for match in matches:
        if '.py' in match:
            continue

        if match == './modules':
            continue
        if match == './tools':
            continue

        if os.path.isdir(match):
            shutil.rmtree(match)
        elif os.path.isfile(match):
            os.unlink(match)
        else:
            pass



