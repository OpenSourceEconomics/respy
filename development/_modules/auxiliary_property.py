from datetime import datetime
import numpy as np
import fileinput
import importlib
import fnmatch
import string
import random
import shutil
import shlex
import glob
import sys
import os

# RESPY directory. This allows to compile_ the debug version of the FORTRAN
# program.
RESPY_DIR = os.path.dirname(os.path.realpath(__file__))
RESPY_DIR = RESPY_DIR.replace('development/testing/property/_modules',
                              '') + 'respy'

PYTHON_EXEC = sys.executable

from respy.python.shared.shared_constants import IS_PARALLEL
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import IS_F2PY


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
    os.system(PYTHON_EXEC + ' waf distclean; ' + PYTHON_EXEC + ' waf configure build --debug')
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
    with open('property.respy.info', 'a') as log_file:
        log_file.write('\n\tRUN COMPLETED\n\n')
        fmt_ = '\t\t{0[0]:<15}{0[1]:>9}\n\n'
        log_file.write(fmt_.format(['TOTAL TESTS', total_tests]))
        log_file.write(fmt_.format(['FAILED TESTS', failed_tests]))


def initialize_record_canvas(full_test_record, start, timeout):
    """ This function initializes the raw log file.
    """
    start_time = start.strftime("%Y-%m-%d %H:%M:%S")
    end_time = (start + timeout).strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open('property.respy.info', 'w') as log_file:
        # Write out some header information.
        log_file.write('\n\n')
        str_ = '\t{0[0]:<15}{0[1]:<20}\n\n'
        log_file.write(str_.format(['START', start_time]))
        log_file.write(str_.format(['FINISH', end_time]))
        log_file.write(str_.format(['UPDATE', current_time]))

        # Note the Python version used during execution.
        str_ = '\t{0[0]:<15}{0[1]}.{0[2]}.{0[3]}\n\n'
        log_file.write(str_.format(['PYTHON'] + list(sys.version_info[:3])))

        log_file.write('\n\n')
        # Iterate over all _modules. There is a potential conflict in the
        # namespace.
        for module_ in full_test_record.keys():
            str_ = '\t{0[0]:<27}{0[1]:<20}{0[2]:<20} \n\n'
            log_file.write(str_.format([module_, 'Success', 'Failure']))
            # Iterate over all methods in the particular module.
            for method_ in sorted(full_test_record[module_]):
                str_ = '\t\t{0[0]:<25}{0[1]:<20}{0[2]:<20} \n'
                success, failure = full_test_record[module_][method_]
                log_file.write(str_.format([method_, success, failure]))

            log_file.write('\n\n')

        log_file.write('-' * 79)
        log_file.write('\n' + '-' * 79)


def update_testing_record(module, method, seed_test, is_success, msg,
        full_test_record):
    """ Maintain a record about the outcomes of the testing efforts.
    """
    # Formatting of time objects.

    str_ = '\t\t{0[0]:<25}{0[1]:<20}{0[2]:<20} \n'
    success, failure = full_test_record[module][method]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_line = '\t{0[0]:<15}{0[1]:<20}\n\n'.format(['UPDATE', current_time])
    is_module, is_method, is_update = False, False, False

    for line in fileinput.input('property.respy.info', inplace=True):

        list_ = shlex.split(line)
        # Skip empty lines
        if len(list_) > 0:
            if list_[0] == module:
                is_module = True
                is_method = False

            if list_[0] == method:
                is_method = True

            is_update = (list_[0] == 'UPDATE')

        if is_method and is_module or is_update:
            if is_update:
                new_line = update_line
            else:
                new_line = str_.format([method, success, failure])
            print(new_line.rstrip())

            is_module, is_update = False, False
        else:
            print(line.rstrip())

    # Append Error message
    if not is_success:
        # Write out the traceback message to file for future inspection.
        with open('property.respy.info', 'a') as log_file:
            str_ = '\nMODULE {0[0]:<25} METHOD {0[1]:<25} SEED: {0[' \
                     '2]:<10} \n\n'
            log_file.write(str_.format([module, method, seed_test]))
            log_file.write(msg)
            log_file.write('\n' + '-' * 79 + '\n\n')


def get_test_dict(test_dir):
    """ This function constructs a dictionary with the recognized test
    _modules as the keys. The corresponding value is a list with all test
    methods inside that module.
    """
    # Process all candidate _modules.
    current_directory = os.getcwd()
    os.chdir(test_dir)
    test_modules = []
    for test_file in glob.glob('test_*.py'):
        test_module = test_file.replace('.py', '')
        test_modules.append(test_module)
    os.chdir(current_directory)

    # Given the _modules, get all tests methods.
    test_dict = dict()
    for test_module in test_modules:
        test_dict[test_module] = []
        mod = importlib.import_module(test_module.replace('.py', ''))
        candidate_methods = dir(mod.TestClass)
        for candidate_method in candidate_methods:
            if 'test_' in candidate_method:
                test_dict[test_module].append(candidate_method)

    # If the PARALLELISM or FORTRAN is not available, we remove the
    # parallel tests.
    if not IS_PARALLEL:
        del test_dict['test_parallelism']

    if not IS_FORTRAN:
        del test_dict['test_versions']
        del test_dict['test_restud']

    if not IS_F2PY:
        del test_dict['test_f2py']

    # The ambiguity tests require special care as they are a collected from
    # different modules, which have their own requirements.
    tests = []
    if not IS_FORTRAN:
        tests += ['test_versions_1', 'test_versions_2', 'test_versions_4']
        tests += ['test_versions_5']
    if not IS_F2PY:
        tests += ['test_2', 'test_3', 'test_4']
        tests += ['test_f2py_4', 'test_f2py_5', 'test_f2py_6']

    for test in tests:
        test_dict['test_ambiguity'].remove(test)

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
        for fname in ['./property.respy.info']:
            if fname in matches:
                matches.remove(fname)

    # Iterate over all remaining files.
    for match in matches:
        if '.py' in match:
            continue

        if match == './_modules':
            continue
        if match == './tools':
            continue

        if os.path.isdir(match):
            shutil.rmtree(match)
        elif os.path.isfile(match):
            os.unlink(match)
        else:
            pass



