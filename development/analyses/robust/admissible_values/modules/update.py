#!/usr/bin/env python
""" This script updates the results from the server.
"""

# standard library
from stat import S_ISDIR

import argparse
import paramiko
import socket
import sys
import os

# project library
from clean import cleanup

# Check for Python 3 and host
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')
if not socket.gethostname() == 'pontos':
    raise AssertionError('Please use @pontos as host')

# module-wide variables
ROBUPY_DIR = os.environ['ROBUPY']
CLIENT_DIR = '/home/eisenhauer/robustToolbox/package/development/analyses' \
             '/drafts/admissible_values'
KEY_DIR = '/home/peisenha/.ssh/id_rsa'

HOST = os.path.dirname(os.path.realpath(__file__)).replace('modules', '')

# Set baseline working directory on HOST
os.chdir(HOST)

''' Functions
'''


def isdir(path, sftp):
    """ Check whether path is a directory.
    """
    try:
        return S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        return False


def _get_remote_material(sftp):
    """ Get all recursively all material from remote SFTP server. The
    function checks for all material in the current remote directory,
    determines the status of file or directory. Files are downloaded
    directly, while directories are iterated through.
    """
    for candidate in sftp.listdir('.'):
        # Determine status of directory or file
        is_directory = isdir(candidate, sftp)
        # Separate Treatment of files and directories
        if not is_directory:
            sftp.get(candidate, candidate)
        else:
            os.mkdir(candidate), os.chdir(candidate), sftp.chdir(candidate)
            _get_remote_material(sftp)
            sftp.chdir('../'), os.chdir('../')


def get_directory(candidate, sftp):
    """ Copy directory from remote to local machine.
    """
    os.mkdir(candidate), os.chdir(candidate), sftp.chdir(candidate)
    _get_remote_material(sftp), sftp.chdir('../'), os.chdir('../')


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    is_all = args.is_all

    # Check arguments
    assert (is_all in [True, False])

    # Finishing
    return is_all


def get_results(is_all):
    """ Get results from server.
    """
    # Starting with clean slate
    cleanup()

    # Read private keys
    key = paramiko.RSAKey.from_private_key_file(KEY_DIR)

    # Open a transport
    transport = paramiko.Transport(('acropolis.uchicago.edu', 22))
    transport.connect(username='eisenhauer', pkey=key)

    # Initialize SFTP connection
    sftp = paramiko.SFTPClient.from_transport(transport)

    # Get files
    sftp.chdir(CLIENT_DIR)

    # Get results directory, this is always downloaded.
    get_directory('rslts', sftp)

    # If requested, also download all intermediate results as well.
    if is_all:
        for candidate in sftp.listdir('.'):
            # Only directories are even potentially interesting.
            if not isdir(candidate, sftp):
                continue
            # Check if directory name can be transformed into a float.
            try:
                float(candidate)
            except ValueError:
                continue
            # Download if
            get_directory(candidate, sftp)

    # Finishing
    sftp.close(), transport.close()


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Download results from @acropolis server.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--all', action='store_true', dest='is_all',
        default=False, help='download all files')

    # Process command line arguments
    is_all = distribute_arguments(parser)

    # Run function
    get_results(is_all)
