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
    levels.sort()

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
                rslts['opt'][level] = (intercept, crit)

    # Write out information.
    with open('indifference_curve.robupy.log', 'w') as out_file:

        # Write out optimal information.
        out_file.write('Optimal\n\n')
        # Formatting for all remaining output.
        fmt = ' {0:<15}{1:<15}{2:<15}\n\n'
        args = ('Level', 'Intercept', 'Criterion')
        # Write out to file.
        out_file.write(fmt.format(*args))
        # Iterate over optimal solution.
        for level in levels:
            # Extract information.
            intercept, criterion = rslts['opt'][level]
            # Formatting for all remaining output.
            fmt = ' {0:<15.3f}{1:<15.3f}{2:<15.3f}\n'
            args = (level, intercept, criterion)
            # Write to file.
            out_file.write(fmt.format(*args))
        out_file.write('\n')

        # Write out all all information.
        out_file.write('All\n\n')
        # Heading
        fmt = ' {0:<15}{1:<15}{2:<15}\n\n'
        args = ('Level', 'Intercept', 'Criterion')
        out_file.write(fmt.format(*args))
        # Iterate over all points.
        for level in levels:
            for i, values in enumerate(rslts[level]):
                # Extract information
                intercept, criterion = values
                # Formatting for all remaining output.
                fmt = ' {0:<15.3f}{1:<15.3f}{2:<15.3f}\n'
                args = (level, intercept, criterion)
                # Write to file.
                out_file.write(fmt.format(*args))
            out_file.write('\n')

    # Store results for further processing.
    os.chdir('../')

    pkl.dump(rslts, open('rslts/indifference_curve.robupy.pkl', 'wb'))


''' Execution of module as script.
'''

if __name__ == '__main__':

    # Process results from result files.
    aggregate()
