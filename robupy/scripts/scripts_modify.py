#!/usr/bin/env python
""" This script allows to modify parameter values.

    Example:

        robupy-modify --fix --identifiers 1-5 5-9

"""

# standard library
import numpy as np

import argparse
import shutil
import os

# project library
from robupy.python.estimate.estimate_auxiliary import dist_optim_paras
from robupy.python.shared.shared_auxiliary import dist_model_paras
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
    action = args.action
    update = args.update

    # Special processing for identifiers to allow to pass in ranges.
    identifiers_list = []
    for identifier in identifiers:
        is_range = ('-' in identifier)
        if is_range:
            identifier = identifier.split('-')
            assert (len(identifier) == 2)
            identifier = [int(val) for val in identifier]
            identifier = list(range(identifier[0], identifier[1] + 1))
        else:
            identifier = [int(identifier)]

        identifiers_list += identifier

    # Check duplicates
    assert (len(set(identifiers_list)) == len(identifiers_list))

    # Checks
    assert os.path.exists(init_file)
    assert os.path.exists('paras_steps.robupy.log')
    assert isinstance(identifiers, list)

    if values is not None:
        assert isinstance(values, list)
        assert (len(values) == len(identifiers_list))
    # Implications
    if action in ['free', 'fix']:
        assert (values is None)
        assert os.path.exists(init_file)
    else:
        assert (update is False)

    # Finishing
    return identifiers_list, values, action, init_file, update


''' Main function
'''


def change_status(identifiers, init_file, is_fixed, update):

    # Baseline
    robupy_obj, init_dict = read(init_file, True)

    if update:
        paras_steps = np.loadtxt(open('paras_steps.robupy.log', 'r'))

        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, _ = \
            dist_optim_paras(paras_steps, True)

    else:
        model_paras = robupy_obj.get_attr('model_paras')
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, _ = \
            dist_model_paras(model_paras, True)

    # Special treatment for covariance matrix
    for identifier in identifiers:
        if identifier >= 16:
            identifier = 16

        if identifier in list(range(0, 6)):
            j = identifier
            init_dict['OCCUPATION A']['coeffs'][j] = coeffs_a[j]
            init_dict['OCCUPATION A']['fixed'][j] = is_fixed
        elif identifier in list(range(6, 12)):
            j = identifier - 6
            init_dict['OCCUPATION B']['coeffs'][j] = coeffs_b[j]
            init_dict['OCCUPATION B']['fixed'][j] = is_fixed
        elif identifier in list(range(12, 15)):
            j = identifier - 12
            init_dict['EDUCATION']['coeffs'][j] = coeffs_edu[j]
            init_dict['EDUCATION']['fixed'][j] = is_fixed
        elif identifier in list(range(15, 16)):
            j = identifier - 15
            init_dict['HOME']['coeffs'][j] = coeffs_home[j]
            init_dict['HOME']['fixed'][j] = is_fixed
        elif identifier in list(range(16, 26)):
            init_dict['SHOCKS']['coeffs'] = shocks_cov
            init_dict['SHOCKS']['fixed'] = np.tile(is_fixed, (4, 4))
        else:
            raise NotImplementedError

        print_random_dict(init_dict)
        shutil.move('test.robupy.ini', init_file)


def change_value(identifiers, init_file, values):
    """ Provide some additional information during estimation run.
    """

    # Read in some baseline information
    robupy_obj = read(init_file)

    paras_fixed = robupy_obj.get_attr('paras_fixed')

    paras_steps = np.genfromtxt('paras_steps.robupy.log')

    # Apply modifications
    for i, identifier in enumerate(identifiers):
        assert (not paras_fixed[identifier])
        paras_steps[identifier] = values[i]

    # Save parametrization to file
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
        nargs='*', default=None, help='updated parameter values', type=float)

    parser.add_argument('--action', action='store', dest='action',
        default=None, help='requested action', type=str, required=True,
        choices=['fix', 'free', 'change'])

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.robupy.ini', help='initialization file')

    parser.add_argument('--update', action='store_true', dest='update',
        default=False, help='update parameter before changing status')


    identifiers, values, action, init_file, update = distribute_input_arguments(
        parser)

    # # Select interface
    if action in ['free', 'fix']:
        is_fixed = (action == 'fix')
        change_status(identifiers, init_file, is_fixed, update)
    elif action in ['change']:
        change_value(identifiers, init_file, values)
