#!/usr/bin/env python
""" Aggregate results from the different subdirectories.
"""

# standard library
import pickle as pkl
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/analyses/robust/_scripts')

# _scripts
from _auxiliary import float_to_string
from _auxiliary import get_float_directories

# project
from auxiliary import get_total_value


def aggregate():
    """ Aggregate results by processing the log files.
    """
    # Dive into results
    os.chdir('rslts')

    # Get all directories that are contain information about the criterion
    # function for selected pairs of ambiguity and psychic costs.
    levels = get_float_directories()
    levels.sort()

    # Iterate over all ambiguity levels and intercepts.
    rslts = dict()
    for level in levels:
        os.chdir(float_to_string(level))
        rslts[level] = []
        # Another process might still be running.
        if os.path.exists('quantification_ambiguity.robupy.log'):
            crit = get_total_value()
            rslts[level] = crit
        # Ready for a new candidate intercept.
        os.chdir('../')

    # Open file for logging purposes.
    with open('quantification_ambiguity.robupy.log', 'w') as out_file:
        # Write out heading information.
        fmt = ' {0:<15}{1:<15}{2:<15}\n\n'
        args = ('Level', 'Value', 'Loss (in %)')
        out_file.write(fmt.format(*args))
        # Iterate over all available levels.
        for level in levels:
            # Get interesting results.
            value, baseline = rslts[level], rslts[0.0]
            difference = ((value / baseline) - 1.0) * 100
            # Format and print string.
            fmt = ' {0:<15.3f}{1:<15.3f}{2:<+15.5f}\n'
            args = (level, value, difference)
            # Write out file.
            out_file.write(fmt.format(*args))
        out_file.write('\n\n')
    # Store results for further processing.
    pkl.dump(rslts, open('quantification_ambiguity.robupy.pkl', 'wb'))
    # Back to root directory.
    os.chdir('../')


''' Execution of module as script.
'''

if __name__ == '__main__':

    # Process results from result files.
    aggregate()
