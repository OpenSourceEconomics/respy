#!/usr/bin/env python
""" This module contains the functions to plot the results exploring the set
of admissible value functions.
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
from auxiliary import plot_admissible_values


''' Core function
'''


def create():
    """ Create a visual representation of all admissible value functions.
    """
    # Read results
    rslts = pkl.load(open('admissible.robupy.pkl', 'rb'))

    # Plot the results from the model misspecification exercise.
    plot_admissible_values(rslts)


''' Execution of module as script.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    create()
