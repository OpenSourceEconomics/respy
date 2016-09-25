#!/usr/bin/env python
""" This script allows to modify parameter values.

    Example:

        respy-modify --fix --identifiers 1-5 5-9

"""
import argparse
import os

from respy.python.estimate.estimate_auxiliary import get_optim_paras
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
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

    # Finishing
    return identifiers_list, action, init_file, values


def scripts_modify(identifiers, action, init_file, values=None):
    """ Modify optimization parameters by either changing their status or
    values.
    """
    # Select interface
    is_fixed = (action == 'fix')

    # Baseline
    init_dict = read(init_file)
    respy_obj = RespyCls(init_file)

    model_paras = respy_obj.get_attr('model_paras')
    level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
            dist_model_paras(model_paras, True)

    x = get_optim_paras(level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                    shocks_cholesky, 'all', None, True)

    # This transformation is necessary as internally the Cholesky
    # decomposition is used but here we operate from the perspective of the
    # initialization file, where the flattened covariance matrix is specified.
    shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)

    # Transform to the external value
    x[0] = x[0] ** 2

    if action == 'value':
        for i, j in enumerate(identifiers):
            x[j] = values[i]

    for identifier in identifiers:
        if identifier in [0]:
            j = identifier
            init_dict['AMBIGUITY']['coeffs'][j] = x[identifier]
            init_dict['AMBIGUITY']['fixed'][j] = is_fixed
        elif identifier in list(range(1, 7)):
            j = identifier - 1
            init_dict['OCCUPATION A']['coeffs'][j] = x[identifier]
            init_dict['OCCUPATION A']['fixed'][j] = is_fixed
        elif identifier in list(range(7, 13)):
            j = identifier - 7
            init_dict['OCCUPATION B']['coeffs'][j] = x[identifier]
            init_dict['OCCUPATION B']['fixed'][j] = is_fixed
        elif identifier in list(range(13, 16)):
            j = identifier - 13
            init_dict['EDUCATION']['coeffs'][j] = x[identifier]
            init_dict['EDUCATION']['fixed'][j] = is_fixed
        elif identifier in list(range(16, 17)):
            j = identifier - 16
            init_dict['HOME']['coeffs'][j] = x[identifier]
            init_dict['HOME']['fixed'][j] = is_fixed
        elif identifier in list(range(17, 27)):
            j = identifier - 17
            init_dict['SHOCKS']['coeffs'][j] = shocks_coeffs[j]
            init_dict['SHOCKS']['fixed'][j] = is_fixed
        else:
            raise NotImplementedError

        # Print dictionary to file
        print_init_dict(init_dict, init_file)

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
