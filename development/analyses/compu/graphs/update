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
CLIENT_DIR = '/home/eisenhauer/robustToolbox/package/development/analyses' \
             '/robust/simulations'
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
    print(CLIENT_DIR + '/rslts')
    sftp.chdir(CLIENT_DIR + '/rslts')

    # Determine available results
    names = []

    for candidate in sftp.listdir('.'):
        try:
            candidate = float(candidate)
        except ValueError:
            continue
        names += ['{0:0.3f}'.format(candidate)]

    # Enter results container
    os.mkdir('rslts'), os.chdir('rslts')

    # Download files
    for name in names:

        os.mkdir(name)

        os.chdir(name)

        sftp.chdir(name)

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