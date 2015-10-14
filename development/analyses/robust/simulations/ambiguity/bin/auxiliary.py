""" This module contains some routines that help in the analysis of policy
responsiveness.
"""

# standard library
from stat import S_ISDIR

import os
import sys
import glob
import shlex
import pickle as pkl

import matplotlib.pylab as plt

sys.path.insert(0, os.environ['ROBUPY'])
from robupy import solve
from robupy import read

# module-wide variables
STYLES = ['--k', '-k', '*k', ':k']
OCCUPATIONS = ['Occupation A', 'Occupation B', 'Schooling', 'Home']
LABELS_SUBSET = ['0.000', '0.010', '0.020']
MAX_PERIOD = 25


""" Auxiliary functions
"""


def solve_ambiguous_economy(level):
    """ Solve an economy in a subdirectory.
    """
    # Process baseline file
    robupy_obj = read('../model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')

    # Formatting directory name
    name = '{0:0.3f}'.format(level)

    # Create directory
    os.mkdir(name)

    os.chdir(name)

    # Update level of ambiguity
    init_dict['AMBIGUITY']['level'] = level

    # Write initialization file
    print_random_dict(init_dict)

    # Solve requested model
    robupy_obj = read('test.robupy.ini')

    solve(robupy_obj)

    # Finishing
    os.chdir('../')


def track_final_choices():
    """ Track the final choices from the ROBUPY output.
    """
    # Auxiliary objects
    levels = get_levels()

    # Process benchmark file
    robupy_obj = read('model.robupy.ini')

    num_periods = robupy_obj.get_attr('num_periods')

    # Create dictionary with the final shares for varying level of ambiguity.
    shares = dict()
    for occu in OCCUPATIONS:
        shares[occu] = []

    # Iterate over all available ambiguity levels
    for level in levels:
        file_name = 'rslts/' + level + '/data.robupy.info'
        with open(file_name, 'r') as output_file:
            for line in output_file.readlines():
                # Split lines
                list_ = shlex.split(line)
                # Skip empty lines
                if not list_:
                    continue
                # Extract shares
                if str(num_periods) in list_[0]:
                    for i, occu in enumerate(OCCUPATIONS):
                        shares[occu] += [float(list_[i + 1])]

    # Finishing
    pkl.dump(shares, open('rslts/ambiguity_shares_final.pkl', 'wb'))


def track_schooling_over_time():
    """ Create dictionary which contains the simulated shares over time for
    varying levels of ambiguity.
    """
    # Auxiliary objects
    levels = get_levels()

    # Iterate over results
    shares = dict()
    for level in levels:
        # Construct dictionary
        shares[level] = dict()
        for choice in OCCUPATIONS:
            shares[level][choice] = []
        # Process results
        file_name = 'rslts/' + level + '/data.robupy.info'
        with open(file_name, 'r') as output_file:
            for line in output_file.readlines():
                # Split lines
                list_ = shlex.split(line)
                # Check relevance
                try:
                    int(list_[0])
                except ValueError:
                    continue
                except IndexError:
                    continue
                # Process line
                for i, occu in enumerate(OCCUPATIONS):
                    shares[level][occu] += [float(list_[i + 1])]

    # Finishing
    pkl.dump(shares, open('rslts/ambiguity_shares_time.pkl', 'wb'))


def get_levels():
    """ Infer ambiguity levels from directory structure.
    """
    os.chdir('rslts')

    levels = []

    for level in glob.glob('*/'):
        # Cleanup strings
        level = level.replace('/', '')
        # Collect levels
        levels += [level]

    os.chdir('../')

    # Finishing
    return sorted(levels)


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


""" Plotting function
"""


def plot_choices_ambiguity(levels, shares_ambiguity):
    """ Plot the choices in the final periods under different levels of
    ambiguity.
    """
    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    # Draw lines
    for i, key_ in enumerate(shares_ambiguity.keys()):
        ax.plot(levels, shares_ambiguity[key_], STYLES[i], label=key_)

    # Both axes
    ax.tick_params(axis='both', right='off', top='off')

    # labels
    ax.set_xlabel('Ambiguity', fontsize=20)
    ax.set_ylabel('Shares', fontsize=20)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=4, fontsize=20)

    # Write out to
    plt.savefig('rslts/choices_ambiguity.png', bbox_inches='tight',
                format='png')


def plot_schooling_ambiguity(shares_time):
    """ Plot schooling level over time for different levels of ambiguity.
    """
    plt.figure(figsize=(12, 8)).add_subplot(111)

    theta = [r'$\theta$', r'$\theta^{\prime}$', r'$\theta^{\prime\prime}$']

    # Initialize plot
    for choice in ['Schooling']:

        # Initialize canvas
        ax = plt.figure(figsize=(12,8)).add_subplot(111)

        # Baseline
        for i, label in enumerate(LABELS_SUBSET):
            ax.plot(range(1, MAX_PERIOD + 1), shares_time[label][choice][
                                         :MAX_PERIOD],
                    label=theta[i], linewidth=5)

        # Both axes
        ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
            right='off')

        # Remove first element on y-axis
        ax.yaxis.get_major_ticks()[0].set_visible(False)

        ax.set_xlim([1, MAX_PERIOD]), ax.set_ylim([0, 0.60])

        # labels
        ax.set_xlabel('Periods', fontsize=16)
        ax.set_ylabel('Share in ' + choice, fontsize=16)

        # Set up legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=False, frameon=False, shadow=False,
            ncol=len(LABELS_SUBSET), fontsize=20)

        file_name = choice.replace(' ', '_').lower() + '_ambiguity.png'

        # Write out to
        plt.savefig('rslts/' + file_name, bbox_inches='tight', format='png')