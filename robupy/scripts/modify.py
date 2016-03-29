#!/usr/bin/env python
""" This script allows to modify parameter values.
"""

# standard library
import numpy as np

import argparse
import os

""" Auxiliary function
"""


def distribute_input_arguments(parser):
    """ Check input for script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    identifiers = args.identifiers
    values = args.values

    # Checks
    assert os.path.exists('paras_steps.robupy.log')
    assert isinstance(identifiers, list)
    assert isinstance(values, list)

    # Finishing
    return identifiers, values


""" Main function
"""


def modify(identifiers, values):
    """ Provide some additional information during estimation run.
    """

    # Read in baseline
    paras_steps = np.genfromtxt('paras_steps.robupy.log')

    # Apply modifications
    for i, j in enumerate(identifiers):
        paras_steps[j] = values[i]
    np.savetxt(open('paras_steps.robupy.log', 'wb'), paras_steps, fmt='%15.8f')


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Modify parameter values for an estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--identifiers', action='store', dest='identifiers',
        nargs='*', default=None, help='parameter identifiers', required=True)

    parser.add_argument('--values', action='store', dest='values',
        nargs='*', default=None, help='updated parameter values', required=True)

    identifiers, values = distribute_input_arguments(parser)

    modify(identifiers, values)
