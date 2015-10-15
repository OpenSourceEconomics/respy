#!/usr/bin/env python
""" This script updates the results from the server.
"""

# standard library
import argparse
import paramiko
import socket
import sys
import os

# Check for Python 3 and host
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')
if not socket.gethostname() == 'pontos':
    raise AssertionError('Please use @pontos as host')


from auxiliary import get_remote_material

# module-wide variables
CLIENT_DIR = '/home/eisenhauer/robustToolbox/package/development/analyses' \
             '/robust/simulations/policy'
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
    os.system('./clean')

    # Read private keys
    key = paramiko.RSAKey.from_private_key_file(KEY_DIR)

    # Open a transport
    transport = paramiko.Transport(('acropolis.uchicago.edu', 22))
    transport.connect(username='eisenhauer', pkey=key )

    # Initialize SFTP connection
    sftp = paramiko.SFTPClient.from_transport(transport)

    # Get files
    sftp.chdir(CLIENT_DIR + '/rslts')

    # Enter results container
    os.mkdir('rslts'), os.chdir('rslts')

    if is_all:
        get_remote_material(sftp)
    else:
        sftp.get('policy_responsiveness.pkl', 'policy_responsiveness.pkl')

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
