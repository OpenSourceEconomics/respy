#!/usr/bin/env python
""" This module contains the functions to plot the results from the model
misspecification exercises.
"""

# standard library
import pickle as pkl

import argparse
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/analyses/robust/_scripts')
sys.path.insert(0, ROBUPY_DIR)

# _scripts
from _auxiliary import float_to_string

# project library
from auxiliary import plot_indifference_curve
from auxiliary import plot_choice_patterns
from auxiliary import get_period_choices

''' Core function
'''


def create():
    """ Create a visual representation of the results from the model
    misspecification exercise.
    """
    # Read results
    rslts = pkl.load(open('rslts/indifference_curve.robupy.pkl', 'rb'))

    # Prepare results for plotting, redo scaling
    levels = sorted(rslts['opt'].keys())
    intercepts = []
    for level in levels:
        intercepts += [rslts['opt'][level][0]]

    # Plot the results from the model misspecification exercise.
    plot_indifference_curve(intercepts, levels)

    # If all detailed information was downloaded, then we can also have a
    # look at the distribution of choices for these economies.
    for level in intercepts:
        if not os.path.exists('rslts/' + float_to_string(level)):
            return

    # Create graphs with choice probabilities over time.
    os.chdir('rslts')

    for level in rslts['opt'].keys():

        # Get the optimal value.
        intercept = rslts['opt'][level][0]

        # Step into optimal subdirectory.
        os.chdir(float_to_string(level))
        os.chdir(float_to_string(intercept))

        # Get the choice probabilities
        choices = get_period_choices()

        # Back to level of rslt directory
        os.chdir('../'), os.chdir('../')

        # Create graphs with choice probabilities over time
        plot_choice_patterns(choices, level)

    os.chdir('../')

''' Execution of module as script.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    create()
