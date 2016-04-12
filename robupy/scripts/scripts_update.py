#!/usr/bin/env python
""" This script allows upgrades the initialization file with the parameter
values from the last step.
"""

# standard library
import numpy as np

import argparse
import os

# project library
from robupy.python.estimate.estimate_auxiliary import dist_optim_paras
from robupy.tests.codes.random_init import print_random_dict
from robupy import read


''' Auxiliary function
'''


def dist_input_arguments(parser):
    """ Check input for script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    init_file = args.init_file

    # Checks
    assert os.path.exists(init_file)
    assert os.path.exists('paras_steps.robupy.log')

    # Finishing
    return init_file


''' Main function
'''


def scripts_update(init_file):
    """ Update model parametrization in initialization file.
    """
    # Collect baseline update
    _, init_dict = read(init_file, True)

    paras_steps = np.genfromtxt('paras_steps.robupy.log')

    # Get and construct ingredients
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov \
        = dist_optim_paras(paras_steps, True)

    shocks_coeff = shocks_cov[np.tril_indices_from(shocks_cov)].tolist()

    # Update initialization dictionary
    init_dict['OCCUPATION A']['coeffs'] = coeffs_a
    init_dict['OCCUPATION B']['coeffs'] = coeffs_b
    init_dict['EDUCATION']['coeffs'] = coeffs_edu
    init_dict['SHOCKS']['coeffs'] = shocks_coeff
    init_dict['HOME']['coeffs'] = coeffs_home

    print_random_dict(init_dict, init_file)

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Update model initialization file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.robupy.ini', help='initialization file')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run update
    scripts_update(args)
