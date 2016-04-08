#!/usr/bin/env python
""" This script allows to modify parameter values.
"""

# standard library
import numpy as np

import argparse
import shutil
import os

# project library
from robupy.python.estimate.estimate_auxiliary import get_model_parameters
from robupy.tests.codes.random_init import print_random_dict
from robupy import read


''' Auxiliary function
'''


def distribute_input_arguments(parser):
    """ Check input for script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    identifiers = args.identifiers
    init_file = args.init_file
    values = args.values
    free = args.free
    fix = args.fix

    # Checks
    assert os.path.exists('paras_steps.robupy.log')
    assert isinstance(identifiers, list)

    if values is not None:
        assert isinstance(values, list)

    # Implications
    if fix or free:
        assert (values is None)
        assert os.path.exists(init_file)
    if fix:
        assert (free is False)
    if free:
        assert (fix is False)

    # Finishing
    return identifiers, values, fix, free, init_file


''' Main function
'''


def change_status(identifiers, fix, free):

    paras_steps = np.loadtxt(open('paras_steps.robupy.log', 'r'))

    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, _ = \
        get_model_parameters(paras_steps, True)

    # Baseline
    robupy_obj, init_dict = read(init_file, True)

    paras_fixed = robupy_obj.get_attr('paras_fixed')

    # Special treatment for covariance matrix
    for identifier in identifiers:

        if identifier >= 16:
            identifier = 16

        # Ensure that action is in fact taken.
        if fix:
            assert (not paras_fixed[identifier])
            paras_fixed[identifier] = True
        elif free:
            assert paras_fixed[identifier]
            paras_fixed[identifier] = False
        else:
            raise NotImplementedError

        if identifier in list(range(0, 6)):
            j = identifier
            init_dict['OCCUPATION A']['coeffs'][j] = coeffs_a[j]
        elif identifier in list(range(6, 12)):
            j = identifier - 6
            init_dict['OCCUPATION B']['coeffs'][j] = coeffs_b[j]
        elif identifier in list(range(12, 15)):
            j = identifier - 12
            init_dict['EDUCATION']['coeffs'][j] = coeffs_edu[j]
        elif identifier in list(range(15, 16)):
            j = identifier - 15
            init_dict['HOME']['coeffs'][j] = coeffs_home[j]
        elif identifier in list(range(16, 26)):
            init_dict['SHOCKS']['coeffs'] = shocks_cov
        else:
            raise NotImplementedError

        # Additional information
        init_dict['ADDITIONAL'] = dict()
        init_dict['ADDITIONAL']['paras_fixed'] = paras_fixed

        print_random_dict(init_dict)
        shutil.move('test.robupy.ini', init_file)


def modify(identifiers, values):
    """ Provide some additional information during estimation run.
    """

    # Read in baseline
    paras_steps = np.genfromtxt('paras_steps.robupy.log')

    # Apply modifications
    for i, j in enumerate(identifiers):
        paras_steps[j] = values[i]

    # Save parametrization to file
    np.savetxt(open('paras_steps.robupy.log', 'wb'), paras_steps, fmt='%15.8f')


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Modify parameter values for an estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--identifiers', action='store', dest='identifiers',
        nargs='*', default=None, help='parameter identifiers', required=True,
        type=int)

    parser.add_argument('--values', action='store', dest='values',
        nargs='*', default=None, help='updated parameter values', type=float)

    parser.add_argument('--fix', action='store_true', dest='fix',
        default=False, help='fix parameters')

    parser.add_argument('--free', action='store_true', dest='free',
        default=False, help='free parameters')

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.robupy.ini', help='initialization file')

    identifiers, values, fix, free, init_file = \
        distribute_input_arguments(parser)

    # Select interface
    if fix or free:
        change_status(identifiers, fix, free)
    else:
        modify(identifiers, values)
