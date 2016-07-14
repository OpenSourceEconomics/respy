import glob
import os
import socket
import subprocess
import sys
from string import Formatter

from config import PACKAGE_DIR
from config import python2_exec
from config import python3_exec

from clsMail import MailCls


def get_executable():
    PYTHON_VERSION = sys.version_info[0]

    if PYTHON_VERSION == 2:
        python_exec = python2_exec
    else:
        python_exec = python3_exec

    return python_exec


def strfdelta(tdelta, fmt):
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
    python_exec = get_executable()
    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR + '/respy')
    subprocess.check_call(python_exec + ' waf distclean', shell=True)
    if not is_debug:
        subprocess.check_call(python_exec + ' waf configure build', shell=True)
    else:
        subprocess.check_call(python_exec + ' waf configure build --debug',
                              shell=True)

    os.chdir(cwd)


def send_notification(which, hours=None, is_failed=False, seed=None,
                      num_tests=None):
    """ Finishing up a run of the testing battery.
    """
    hostname = socket.gethostname()

    if which == 'scalability':
        subject = ' RESPY: Scalability Testing'
        message = ' Scalability testing is completed on @' + hostname + '.'
    elif which == 'reliability':
        subject = ' RESPY: Reliability Testing'
        message = ' Reliability testing is completed on @' + hostname + '.'
    elif which == 'property':
        subject = ' RESPY: Property Testing'
        message = ' A ' + str(hours) + ' hour run of the testing battery on @' + \
                  hostname + ' is completed.'

    elif which == 'release':
        subject = ' RESPY: Release Testing'
        if is_failed:
            message = ' Failure during release testing with seeed ' + str(seed) + '.'
        else:
            message = ' Release testing completed successfully after ' + str(
                hours) + ' hours. We ran a total of ' + str(num_tests) + \
                      ' tests.'
    else:
        raise AssertionError

    mail_obj = MailCls()
    mail_obj.set_attr('subject', subject)
    mail_obj.set_attr('message', message)

    if which == 'property':
        mail_obj.set_attr('attachment', 'property.respy.info')
    elif which == 'scalability':
        mail_obj.set_attr('attachment', 'scalability.respy.info')

    mail_obj.lock()

    mail_obj.send()


def aggregate_information(which):

    if which == 'scalability':
        fname_info = 'scalability.respy.info'
    elif which == 'reliability':
        fname_info = 'reliability.respy.info'
    else:
        raise AssertionError


    dirnames = []
    for fname in next(os.walk('.'))[1]:
        dirnames += [fname.replace('.ini', '')]
    with open(fname_info, 'w') as outfile:
        outfile.write('\n')
        for dirname in dirnames:
            outfile.write(' ' + dirname + '\n')
            os.chdir(dirname)
            with open(fname_info, 'r') as infile:
                outfile.write(infile.read())
            os.chdir('../')
            outfile.write('\n\n')