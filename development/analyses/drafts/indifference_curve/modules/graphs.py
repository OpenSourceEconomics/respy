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
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR)

# project library
from auxiliary import plot_model_misspecification


''' Core function
'''


def create():
    """ Create a visual representation of the results from the model
    misspecification exercise.
    """
    # Read results
    rslts = pkl.load(open('rslts/model_misspecification.robupy.pkl', 'rb'))

    # Prepare results for plotting, redo scaling
    xvals = sorted(rslts.keys())
    yvals = []
    for value in xvals:
        yvals += [rslts[value] * 100000.00]

    # Plot the results from the model misspecification exercise.
    plot_model_misspecification(yvals, xvals)


''' Execution of module as script.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    create()
