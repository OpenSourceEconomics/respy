from string import Formatter

import subprocess
import socket
import sys
import os

import numpy as np

from config import PACKAGE_DIR
from config import python2_exec
from config import python3_exec

from clsMail import MailCls

from config import SPECS


def update_class_instance(respy_obj, spec_dict):
    """ Update model specification from the baseline initialization file.
    """

    respy_obj.unlock()

    # Varying the baseline level of ambiguity requires special case.
    if 'level' in spec_dict['update'].keys():
        respy_obj.attr['optim_paras']['level'] = np.array([spec_dict['update']['level']])

    for key_ in spec_dict['update'].keys():
        if key_ in ['level']:
            continue
        respy_obj.set_attr(key_, spec_dict['update'][key_])

    respy_obj.lock()

    return respy_obj


def get_executable():

    PYTHON_VERSION = sys.version_info[0]

    if PYTHON_VERSION == 2:
        python_exec = python2_exec
    else:
        python_exec = python3_exec

    return python_exec


def strfdelta(tdelta, fmt):
    """ Get a string from a timedelta.
    """
    f, d = Formatter(), {}
    l = {'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    k = list(map(lambda x: x[1], list(f.parse(fmt))))
    rem = int(tdelta.total_seconds())
    for i in ('D', 'H', 'M', 'S'):
        if i in k and i in l.keys():
            d[i], rem = divmod(rem, l[i])
    return f.format(fmt, **d)


def cleanup():
    os.system('git clean -d -f')

def compile_package(is_debug=False):
    """ Compile the package for use.
    """
    python_exec = get_executable()
    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR + '/respy')
    subprocess.check_call(python_exec + ' waf distclean', shell=True)
    if not is_debug:
        subprocess.check_call(python_exec + ' waf configure build',
            shell=True)
    else:
        subprocess.check_call(python_exec + ' waf configure build --debug ',
            shell=True)

    os.chdir(cwd)


def send_notification(which, **kwargs):
    """ Finishing up a run of the testing battery.
    """

    # This allows to run the scripts even when no notification can be send.
    if not os.path.exists(os.environ['HOME'] + '/.credentials'):
        return

    hours, is_failed, num_tests, seed = None, None, None, None

    # Distribute keyword arguments
    if 'is_failed' in kwargs.keys():
        is_failed = kwargs['is_failed']

    if 'hours' in kwargs.keys():
        hours = '{}'.format(kwargs['hours'])

    if 'num_tests' in kwargs.keys():
        num_tests = '{}'.format(kwargs['num_tests'])

    if 'seed' in kwargs.keys():
        seed = '{}'.format(kwargs['seed'])

    if 'test_idx' in kwargs.keys():
        test_idx = '{}'.format(kwargs['test_idx'])

    if 'old_release' in kwargs.keys():
        old_release = kwargs['old_release']

    if 'new_release' in kwargs.keys():
        new_release = kwargs['new_release']

    hostname = socket.gethostname()

    if which == 'scalability':
        subject = ' RESPY: Scalability Testing'
        message = ' Scalability testing is completed on @' + hostname + '.'
    elif which == 'reliability':
        subject = ' RESPY: Reliability Testing'
        message = ' Reliability testing is completed on @' + hostname + '.'
    elif which == 'property':
        subject = ' RESPY: Property Testing'
        message = ' A ' + hours + ' hour run of the testing battery on @' + \
                  hostname + ' is completed.'

    elif which == 'regression':
        subject = ' RESPY: Regression Testing'
        if is_failed:
            message = 'Failure during regression testing for test ' + \
                      test_idx + '.'
        else:
            message = ' Regression testing is completed on @' + hostname + '.'

    elif which == 'release':
        subject = ' RESPY: Release Testing'
        if is_failed:
            message = ' Failure during release testing with seed ' + \
                seed + ' on @' + hostname + '.'
        else:
            message = ' Release testing completed successfully after ' + \
                hours + ' hours on @' + hostname + '. We compared release ' + \
                old_release + ' against ' + new_release + ' for a total of ' + \
                num_tests + ' tests.'
    else:
        raise AssertionError

    mail_obj = MailCls()
    mail_obj.set_attr('subject', subject)
    mail_obj.set_attr('message', message)

    if which == 'property':
        mail_obj.set_attr('attachment', 'property.respy.info')
    elif which == 'scalability':
        mail_obj.set_attr('attachment', 'scalability.respy.info')
    elif which == 'reliability':
        mail_obj.set_attr('attachment', 'reliability.respy.info')

    mail_obj.lock()

    mail_obj.send()


def aggregate_information(which):

    if which == 'scalability':
        fname_info = 'scalability.respy.info'
    elif which == 'reliability':
        fname_info = 'reliability.respy.info'
    else:
        raise AssertionError


    dirnames =SPECS


    with open(fname_info, 'w') as outfile:
        outfile.write('\n')
        for dirname in dirnames:

            if not os.path.exists(dirname):
                continue

            outfile.write(' ' + dirname + '\n')
            os.chdir(dirname)
            with open(fname_info, 'r') as infile:
                outfile.write(infile.read())
            os.chdir('../')
            outfile.write('\n\n')
