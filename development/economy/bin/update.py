#!/usr/bin/env python
""" This script updates the results from the server.
"""

# standard library
import argparse
import paramiko
import os

# project library
from clean import cleanup

# module-wide variables
CLIENT_DIR = '/home/eisenhauer/robustToolbox/package/development/economy'
KEY_DIR = '/home/peisenha/.ssh/id_rsa'

''' Functions
'''


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
    transport.connect(username='eisenhauer', pkey=key )

    # Initialize SFTP connection
    sftp = paramiko.SFTPClient.from_transport(transport)

    # Get files
    sftp.chdir(CLIENT_DIR)

    # Determine available results
    levels = []

    for candidate in sftp.listdir('.'):
        try:
            candidate = float(candidate)
        except ValueError:
            continue
        levels += [str(candidate)]

    # Download files
    for level in levels:

        os.mkdir(level)

        os.chdir(level)

        sftp.chdir(level)

        # Select files
        files = ['data.robupy.info']
        if is_all:
            files = sftp.listdir('.')

        # Get files
        for file_ in files:
            sftp.get(file_, file_)

        sftp.chdir('../')

        os.chdir('../')

    # Finishing
    sftp.close()

    transport.close()


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