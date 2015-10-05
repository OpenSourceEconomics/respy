#!/usr/bin/env python
""" This module creates some graphs for the economy in the case of ambiguity.
"""
# standard library
import numpy as np

import shlex
import glob
import sys
import os

import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'])


HOME = os.environ['ROBUPY'] + '/development/analyses/ectra/graphs'

# project package
from robupy import read


robupy_obj = read('../simulations/model.robupy.ini')

num_periods = robupy_obj.get_attr('num_periods')

try:
    os.mkdir('rslts')
except:
    pass


def get_ambiguity_levels():
    """ Infer ambiguity levels from directory structure.
    """

    os.chdir(HOME)

    os.chdir('../simulations/rslts')

    levels = []

    for level in glob.glob('*/'):

        # Cleanup strings
        level = level.replace('/', '')

        # Collect levels
        levels += [level]

    os.chdir(HOME)

    # Finishing
    return sorted(levels)


# Get all ambiguity levels
levels = get_ambiguity_levels()

# Read choice distribution in final state by ambiguity level
choices = ['Occupation A', 'Occupation B', 'Schooling', 'Home']

# Create dictionary with the final shares for varying level of ambiguity.
shares_ambiguity = dict()
for choice in choices:
    shares_ambiguity[choice] = []

for level in levels:
    with open('../simulations/rslts/' + level + '/data.robupy.info', 'r') as output_file:
        for line in output_file.readlines():
            # Split lines
            list_ = shlex.split(line)
            # Skip empty lines
            if not list_:
                continue
            # Extract shares
            if (str(num_periods) in list_[0]):
                for i, choice in enumerate(choices):
                    shares_ambiguity[choice] += [float(list_[i + 1])]

# Create dictionary which contains the simulated shares over time for
# varying levels of ambiguity.
shares_time = dict()

for level in levels:
    # Construct dictionary
    shares_time[level] = dict()
    for choice in choices:
        shares_time[level][choice] = []
    # Process results
    with open('../simulations/rslts/' + level + '/data.robupy.info', 'r') as \
            output_file:
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
            for i, choice in enumerate(choices):
                shares_time[level][choice] += [float(list_[i + 1])]

# Initialize plot
ax = plt.figure(figsize=(12,8)).add_subplot(111)

styles = ['--k', '-k', '*k', ':k']
# Draw lines
for i, key_ in enumerate(shares_ambiguity.keys()):
    ax.plot(levels, shares_ambiguity[key_], styles[i], label=key_)

# Both axes
ax.tick_params(axis='both', right='off', top='off')

# labels
ax.set_xlabel('Ambiguity', fontsize=20)
ax.set_ylabel('Shares', fontsize=20)

# Set up legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
    fancybox=False, frameon=False, shadow=False, ncol=4,
    fontsize=20)

# Write out to
plt.savefig('rslts/choices_ambiguity.pdf', bbox_inches='tight', format="pdf")

# Here I investigate the evolution of schooling over time for different
# levels of ambiguity
labels_subset = ['0.000', '0.010', '0.020']
max_period = 25

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
    ax.tick_params(axis='both', right='off', top='off')

    # Remove first element on y-axis
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    ax.set_xlim([1, max_period]), ax.set_ylim([0, 0.60])

    # labels
    ax.set_xlabel('Periods', fontsize=20)
    ax.set_ylabel('Share in ' + choice, fontsize=20)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=len(labels_subset),
        fontsize=20)

    file_name = choice.replace(' ', '_').lower() + '_ambiguity.pdf'

    # Write out to
    plt.savefig('rslts/' + file_name, bbox_inches='tight', format='pdf')