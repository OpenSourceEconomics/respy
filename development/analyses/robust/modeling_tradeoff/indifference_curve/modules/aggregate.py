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
sys.path.insert(0, ROBUPY_DIR + '/development/analyses/robust/_scripts')

# _scripts
from _auxiliary import float_to_string

# project
from auxiliary import get_criterion_function
from auxiliary import get_float_directories


def aggregate():
    """ Aggregate results by processing the log files.
    """
    # Dive into results
    os.chdir('rslts')

    # Get all directories that are contain information about the criterion
    # function for selected pairs of ambiguity and psychic costs.
    levels = get_float_directories()

    # Iterate over all ambiguity levels and intercepts.
    rslts = dict()
    for level in levels:
        os.chdir(float_to_string(level))
        rslts[level] = []
        # For given level of ambiguity, get the intercepts.
        intercepts = get_float_directories()
        for intercept in intercepts:
            os.chdir(float_to_string(intercept))
            # Another process might still be running.
            if os.path.exists('indifference_curve.robupy.log'):
                crit = get_criterion_function()
                rslts[level] += [(intercept, crit)]
            # Ready for a new candidate intercept.
            os.chdir('../')
        # Back to root directory
        os.chdir('../')

    # Construct extra key with optimal results to ease further processing.
    rslts['opt'] = dict()
    for level in levels:
        # Initialize loop
        crit_opt, candidates = np.inf, rslts[level]
        # Iterate over all candidate pairs and update if better pair found.
        for candidate in candidates:
            intercept, crit = candidate
            is_update = crit < crit_opt
            if is_update:
                crit_opt = crit
                rslts['opt'][level] = intercept

    # Back to root directory.
    os.chdir('../')

    # Finishing
    return rslts


''' Execution of module as script.
'''

if __name__ == '__main__':

    # Process results from result files.
    rslts = aggregate()

    # Store results for further processing.
    pkl.dump(rslts, open('rslts/indifference_curve.robupy.pkl', 'wb'))