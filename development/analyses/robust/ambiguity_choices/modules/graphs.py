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
from auxiliary import plot_schooling_ambiguity
from auxiliary import plot_choices_ambiguity


''' Core function
'''


def create():
    """ Create a visual representation of all admissible value functions.
    """

    rslt = pkl.load(open('rslts/ambiguity_choices.robupy.pkl', 'rb'))
    plot_choices_ambiguity(rslt)

    plot_schooling_ambiguity(rslt)


''' Execution of module as script.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    create()
