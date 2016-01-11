#!/usr/bin/env python
""" Aggregate results from the different subdirectories.
"""
# standard library
import pickle as pkl
import numpy as np
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# project
from auxiliary import get_criterion_function
from auxiliary import get_float_directories
from auxiliary import get_name


def aggregate():
    """ Aggregate results.
    """

    levels = get_float_directories()

    rslts = dict()

    for level in levels:

        os.chdir(get_name(level))

        rslts[level] = []

        intercepts = get_float_directories()

        for intercept in intercepts:

            os.chdir(get_name(intercept))

            crit = get_criterion_function()

            rslts[level] += [(intercept, crit)]

            os.chdir('../')

        os.chdir('../')

    # Construct extra key with optimal results.
    rslts['opt'] = dict()
    for level in levels:
        rslts['opt'][level] = np.inf
        candidates = rslts[level]
        for candidate in candidates:
            intercept, crit = candidate
            if crit < rslts['opt'][level]:
                rslts['opt'][level] = intercept

    # Finishing
    return rslts


''' Execution of module as script.
'''

if __name__ == '__main__':

    # Process results from result files.
    rslts = aggregate()

    # Create directory if not existing.
    if not os.path.exists('rslts'):
        os.mkdir('rslts')

    # Store results for further processing.
    pkl.dump(rslts, open('rslts/indifference_curve.robupy.pkl', 'wb'))