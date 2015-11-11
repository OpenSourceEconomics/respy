""" This module contains some routines that help in the analysis of policy
responsiveness.
"""

# standard library
import matplotlib
matplotlib.use('Agg')

import matplotlib.pylab as plt
import numpy as np

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


def plot_policy_responsiveness(rslt):
    """ Plot policy responsivenss.
    """
    # Scaling, percentage increase relative to baseline
    for key_ in [0.00, 0.01, 0.02]:
        rslt[key_][:] = [x / rslt[key_][-1] for x in rslt[key_]]

    # Collect relevant subset
    response = []
    for key_ in [0.00, 0.01, 0.02]:
        response += [(rslt[key_][-2] - 1.00) * 100]

    ax = plt.figure(figsize=(1.05*12, 1.05*8)).add_subplot(111)

    bar_width = 0.35
    plt.bar(np.arange(3) + 0.5 * bar_width, response, bar_width,
        color=['red', 'orange', 'blue'])

    # x axis
    ax.set_xlabel('Level of Ambiguity', fontsize=16)

    # y axis
    ax.set_ylim([0, 4])
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_ylabel('Increase in Average Schooling (in %)', fontsize=16)

    plt.xticks(np.arange(3) + bar_width, ('0.00', '0.01', '0.02'))

    plt.savefig('rslts/policy_responsiveness.png', bbox_inches='tight',
                format='png')
