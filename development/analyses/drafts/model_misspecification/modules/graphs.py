#!/usr/bin/env python
""" This module contains the functions to plot the results from the model
misspecification exercises.
"""

# standard library
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
from auxiliary import plot_model_misspecification


''' Core function
'''


def create():
    """ Create a visual representation of the results from the model
    misspecification exercise.
    """
    # Pull in the baseline specification from the RESTUD directory.
    shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')

    # Get all available results.
    directories = dict()
    directories['float'], directories['string'] = [], []
    for candidate in next(os.walk('.'))[1]:
        # Check if directory can be transformed into a float.
        try:
            float(candidate)
        except ValueError:
            continue
        # Collect the two types
        directories['float'] += [float(candidate)]
        directories['string'] += [candidate]
    # Iterate over all available results directory.
    rslts = dict()
    for dir_ in directories['string']:
        # Enter directory
        os.chdir(dir_)
        # Process result file
        with open('misspecification.robupy.log', 'r') as file_:
            for line in file_.readlines():
                # Split line
                list_ = shlex.split(line)
                # Skip empty lines
                if not list_:
                    continue
                # Check for relevant keyword
                is_result = (list_[0] == 'Result')
                if not is_result:
                    continue
                # Process results
                rslts[dir_] = float(list_[1])
        # Finishing
        os.chdir('../')
    # Prepare directory structure to store the resulting *.png files.
    if os.path.exists('rslts'):
        shutil.rmtree('rslts')
    os.mkdir('rslts')
    # Extract the values from the results dictionary to ensure the right order.
    intercepts = []
    for str_ in sorted(directories['string']):
        intercepts += [rslts[str_]]
    # Plot the results from the model misspecification exercise.
    plot_model_misspecification(sorted(directories['float']), intercepts)

''' Execution of module as script.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    create()
