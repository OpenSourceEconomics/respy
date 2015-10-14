""" This module contains some routines that help in the analysis of policy
responsiveness.
"""

# standard library
from stat import S_ISDIR

import os


def isdir(path, sftp):
    """ Check whether path is a directory.
    """
    try:
        return S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        return False


def formatting(arg):
    """ Formatting of float input to standardized string.
    """
    return '{0:0.3f}'.format(arg)


def get_remote_material(sftp):
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
            get_remote_material(sftp)
            sftp.chdir('../'), os.chdir('../')
