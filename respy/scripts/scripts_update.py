#!/usr/bin/env python
""" This script allows upgrades the initialization file with the parameter
values from the last step.
"""
import argparse
import os

from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_auxiliary import get_est_info
from respy.python.read.read_python import read


def dist_input_arguments(parser):
    """ Check input for script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    init_file = args.init_file

    # Checks
    assert os.path.exists(init_file)
    assert os.path.exists('est.respy.info')

    # Finishing
    return init_file


def scripts_update(init_file):
    """ Update model parametrization in initialization file.
    """
    # Collect baseline update
    init_dict = read(init_file)

    paras_steps = get_est_info()['paras_step']

    # Get and construct ingredients
    level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, _ \
        = dist_optim_paras(paras_steps, True)
    shocks_coeffs = paras_steps[17:]

    # Update initialization dictionary
    init_dict['AMBIGUITY']['coeffs'] = level
    init_dict['OCCUPATION A']['coeffs'] = coeffs_a
    init_dict['OCCUPATION B']['coeffs'] = coeffs_b
    init_dict['EDUCATION']['coeffs'] = coeffs_edu
    init_dict['SHOCKS']['coeffs'] = shocks_coeffs
    init_dict['HOME']['coeffs'] = coeffs_home

    print_init_dict(init_dict, init_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Update model initialization file with parameter values from last '
        'step.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.respy.ini', help='initialization file')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run update
    scripts_update(args)
