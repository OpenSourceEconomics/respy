#!/usr/bin/env python
""" This module contains the functions to plot the results
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
from auxiliary import plot_lifetime_value


''' Core function
'''


def create():
    """ Create a visual representation of ambiguity quantification
    """
    # Read results
    rslts = pkl.load(open('rslts/ambiguity_quantification.robupy.pkl', 'rb'))

    # Plot the results from the model misspecification exercise.
    plot_lifetime_value(rslts)


''' Execution of module as script.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess ambiguity quantification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    create()
