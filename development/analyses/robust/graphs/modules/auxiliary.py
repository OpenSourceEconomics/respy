""" This module contains some auxiliary functions to create the graphs for
the ECTRA economy.
"""

# standard library
import matplotlib.pylab as plt

import shlex
import glob
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'])

# project package
from robupy import read

# module-wide variables
OCCUPATIONS = ['Occupation A', 'Occupation B', 'Schooling', 'Home']
STYLES = ['--k', '-k', '*k', ':k']

""" Auxiliary function
"""


def get_ambiguity_levels():
    """ Infer ambiguity levels from directory structure.
    """
    os.chdir('../simulations/rslts')

    levels = []

    for level in glob.glob('*/'):

        # Cleanup strings
        level = level.replace('/', '')

        # Collect levels
        levels += [level]

    os.chdir('../../graphs')

    # Finishing
    return sorted(levels)


def track_final_choices(levels):
    """ Track the final choices from the ROBUPY output.
    """
    robupy_obj = read('../simulations/model.robupy.ini')

    num_periods = robupy_obj.get_attr('num_periods')

    # Create dictionary with the final shares for varying level of ambiguity.
    shares_ambiguity = dict()
    for occu in OCCUPATIONS:
        shares_ambiguity[occu] = []

    # Iterate over all available ambiguity levels
    for level in levels:
        file_name = '../simulations/rslts/' + level + '/data.robupy.info'
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
                        shares_ambiguity[occu] += [float(list_[i + 1])]

    # Finishing
    return shares_ambiguity


def track_schooling_over_time(levels):
    """ Create dictionary which contains the simulated shares over time for
    varying levels of ambiguity.
    """
    shares_time = dict()

    for level in levels:
        # Construct dictionary
        shares_time[level] = dict()
        for choice in OCCUPATIONS:
            shares_time[level][choice] = []
        # Process results
        file_name = '../simulations/rslts/' + level + '/data.robupy.info'
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
                    shares_time[level][occu] += [float(list_[i + 1])]

    # Finishing
    return shares_time


""" Plotting functions
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


def plot_schooling_ambiguity(labels_subset, max_period, shares_time):
    """ Plot schooling level over time for different levels of ambiguity.
    """
    plt.figure(figsize=(12, 8)).add_subplot(111)

    theta = [r'$\theta$', r'$\theta^{\prime}$', r'$\theta^{\prime\prime}$']

    # Initialize plot
    for choice in ['Schooling']:

        # Initialize canvas
        ax = plt.figure(figsize=(12,8)).add_subplot(111)

        # Baseline
        for i, label in enumerate(labels_subset):
            ax.plot(range(1, max_period + 1), shares_time[label][choice][:max_period],
                    label=theta[i], linewidth=5)

        # Both axes
        ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
            right='off')

        # Remove first element on y-axis
        ax.yaxis.get_major_ticks()[0].set_visible(False)

        ax.set_xlim([1, max_period]), ax.set_ylim([0, 0.60])

        # labels
        ax.set_xlabel('Periods', fontsize=16)
        ax.set_ylabel('Share in ' + choice, fontsize=16)

        # Set up legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=False, frameon=False, shadow=False,
            ncol=len(labels_subset), fontsize=20)

        file_name = choice.replace(' ', '_').lower() + '_ambiguity.png'

        # Write out to
        plt.savefig('rslts/' + file_name, bbox_inches='tight', format='png')