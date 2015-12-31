#!/usr/bin/env python
""" This module contains the functions to plot the results exploring the set
of admissible value functions.
"""

# standard library
import pickle as pkl

import argparse
import shutil
import shlex
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR)

# project library
from auxiliary import plot_admissible_values


''' Core function
'''


def create():
    """ Create a visual representation of the results from the model
    misspecification exercise.
    """
    # Read results
    rslts = pkl.load(open('admissible.robupy.pkl', 'rb'))

    # Prepare results for plotting, redo scaling
    xvals= sorted(rslts.keys())

    # TODO: Getting readdy
#    rslts[0.00] = 0.00
#    rslts[0.01] = -0.03
#    rslts[0.02] = -0.05
#    rslts[0.03] = -0.06

#    yvals = []
#    for value in xvals:
#        yvals += [rslts[value] ]

    # Plot the results from the model misspecification exercise.
    print(rslts)
    plot_admissible_values()


''' Execution of module as script.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    create()
