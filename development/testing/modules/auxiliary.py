
# standard library
import numpy as np

import importlib
import fnmatch
import socket
import shutil
import glob
import os

# testing library
from modules.clsMail import MailCls

''' Auxiliary functions
'''


def finalize_testing_record():
    """ Add the temporary file with the information about tracebacks to the
    main report.
    """
    # Indicate that test run is finished
    with open('report.testing.log', 'a') as log_file:
        log_file.write('   RUN COMPLETED\n\n')

    # Aggregate information from temporary files.
    if os.path.exists('tracebacks.testing.tmp'):

        with open('report.testing.log', 'a') as outfile:
            outfile.write('\n---------------------------------------'
                            '-----------------------------------------\n\n')
            for line in open('tracebacks.testing.tmp'):
                outfile.write(line)
            os.unlink('tracebacks.testing.tmp')


def update_testing_record(module, method, seed, is_success, msg,
        full_test_record, start, timeout):
    """ Maintain a record about the outcomes of the testing efforts.
    """
    # Formatting of time objects.
    start_time = start.strftime("%Y-%m-%d %H:%M:%S")
    end_time = (start + timeout).strftime("%Y-%m-%d %H:%M:%S")

    # Write out overview information such as the number of successful and
    # failed test runs.
    with open('report.testing.log', 'w') as log_file:
        # Write out some header information.
        log_file.write('\n\n')
        string = '\t{0[0]:<15}{0[1]:<20}\n\n'
        log_file.write(string.format(['START', start_time]))
        log_file.write(string.format(['FINISH', end_time]))
        log_file.write('\n\n')
        # Iterate over all modules.
        for module in full_test_record.keys():
            string = '   {0[0]:<25}{0[1]:<20}{0[2]:<20} \n\n'
            log_file.write(string.format([module, 'Success', 'Failure']))
            # Iterate over all methods in the particular module.
            for method in sorted(full_test_record[module]):
                string = '\t{0[0]:<25}{0[1]:<20}{0[2]:<20} \n'
                success, failure = full_test_record[module][method]
                log_file.write(string.format([method, success, failure]))

            log_file.write('\n\n')

    # Special care for failures. However, I need to make sure that the file
    # exists.
    if not is_success:
        # Write out the traceback message to file for future inspection.
        with open('tracebacks.testing.tmp', 'a') as log_file:
            string = 'MODULE {0[0]:<25} METHOD {0[1]:<25} SEED: {0[' \
                     '2]:<10} \n\n'
            log_file.write(string.format([module, method, seed]))
            log_file.write(msg)
            log_file.write('\n---------------------------------------'
                           '-----------------------------------------\n\n')


def get_test_dict(TEST_DIR):
    """ This function constructs a dictionary with the recognized test
    modules as the keys. The corresponding value is a list with all test
    methods inside that module.
    """
    # Process all candidate modules.
    current_directory = os.getcwd()
    os.chdir(TEST_DIR)
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
    index = np.random.random_integers(0, len(candidates) - 1)
    module, method = candidates[index]

    # Finishing
    return module, method


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
        # Check that the credentials file is stored in the user's
        # HOME directory.
        assert (os.path.exists(os.environ['HOME'] + '/.credentials'))

    # Finishing.
    return hours, notification


def send_notification(hours):
    """ Finishing up a run of the testing battery.
    """
    # Auxiliary objects.
    hostname = socket.gethostname()

    subject = ' ROBUPY: Completed Testing Battery '

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
            try:
                matches.remove(fname)
            except Exception:
                pass

    # Iterate over all remaining files.
    for match in matches:
        if '.py' in match:
            continue

        if match == './modules':
            continue

        if os.path.isdir(match):
            shutil.rmtree(match)
        elif os.path.isfile(match):
            os.unlink(match)
        else:
            pass



