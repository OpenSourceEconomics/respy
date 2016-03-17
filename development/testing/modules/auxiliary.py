
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
    index = np.random.random_integers(0, len(candidates))
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


def cleanup(keep_results):
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
    for match in matches:
        if '.py' in match:
            continue
        if match == './modules':
            continue
        if keep_results:
            if match == './logging.log':
                continue

        remove(match)


def remove(name):
    """ This function allows to remove files or directories.
    """
    try:
        os.unlink(name)
    except IsADirectoryError:
        shutil.rmtree(name)

