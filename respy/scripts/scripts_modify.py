#!/usr/bin/env python
""" This script allows to modify parameter values.

    Example:

        respy-modify --fix --identifiers 1-5 5-9

"""

# standard library
import numpy as np

import argparse
import os

# project library
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.read.read_python import read

from respy import RespyCls


def dist_input_arguments(parser):
    """ Check input for script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    identifiers = args.identifiers
    init_file = args.init_file
    values = args.values
    action = args.action

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
    assert isinstance(identifiers, list)

    if values is not None:
        assert isinstance(values, list)
        assert (len(values) == len(identifiers_list))
    # Implications
    if action in ['free', 'fix']:
        assert (values is None)
        assert os.path.exists(init_file)
    elif action in ['change']:
        assert os.path.exists('paras_steps.respy.log')

    # Finishing
    return identifiers_list, values, action, init_file


def scripts_modify(identifiers, values, action, init_file):
    """ Modify optimization parameters by either changing their status or
    values.
    """
    # Select interface
    if action in ['free', 'fix']:
        is_fixed = (action == 'fix')
        change_status(identifiers, init_file, is_fixed)
    elif action in ['value']:
        change_value(identifiers, values)


def change_status(identifiers, init_file, is_fixed):
    """ Change the status of the a list of parameters.
    """
    # Baseline
    init_dict = read(init_file)
    respy_obj = RespyCls(init_file)

    model_paras = respy_obj.get_attr('model_paras')
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, _ = \
            dist_model_paras(model_paras, True)

    for identifier in identifiers:
        # Special treatment for covariance matrix
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
            shocks_coeffs = shocks_cov[np.triu_indices_from(
                shocks_cov)].tolist()
            for i in [0, 4, 7, 9]:
                shocks_coeffs[i] = np.sqrt(shocks_coeffs[i])
            init_dict['SHOCKS']['coeffs'] = shocks_coeffs
            init_dict['SHOCKS']['fixed'] = np.tile(is_fixed, 1)
        else:
            raise NotImplementedError

        # Print dictionary to file
        print_init_dict(init_dict, init_file)


def change_value(identifiers, values):
    """ Provide some additional information during estimation run.
    """

    # Read in some baseline information
    paras_steps = np.genfromtxt('paras_steps.respy.log')

    # Apply modifications
    for i, identifier in enumerate(identifiers):
        paras_steps[identifier] = values[i]

    # Save parametrization to file
    np.savetxt(open('paras_steps.respy.log', 'wb'), paras_steps, fmt='%15.8f')


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
        choices=['fix', 'free', 'value'])

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.respy.ini', help='initialization file')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run modifications
    scripts_modify(*args)
