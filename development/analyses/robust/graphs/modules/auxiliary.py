""" This module contains some auxiliary functions to create the graphs for
the ROBUST economy.
"""

# standard library
import matplotlib.pylab as plt

import shlex
import glob
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'])

# module-wide variables
STYLES = ['--k', '-k', '*k', ':k']

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