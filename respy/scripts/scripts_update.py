#!/usr/bin/env python
""" This script allows upgrades the initialization file with the parameter values from the last
step.
"""
import argparse
import shutil
import os

from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.pre_processing.model_processing import write_init_file
from respy.python.shared.shared_auxiliary import get_est_info
from respy.pre_processing.model_processing import read_init_file
from respy.custom_exceptions import UserError


def dist_input_arguments(parser):
    """ Check input for script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    init_file = args.init_file

    # Checks
    if not os.path.exists(init_file):
        raise UserError('Initialization file does not exist')
    if not os.path.exists('est.respy.info'):
        raise UserError('Information on parameter values from last step unavailable')

    # Finishing
    return init_file


def scripts_update(init_file):
    """ Update model parametrization in initialization file.
    """
    # Collect baseline update
    init_dict = read_init_file(init_file)

    paras_steps = get_est_info()['paras_step']

    # While sometimes useful, we cannot use this script if there are missing values in the
    # parameters due to too large values.
    if '---' in paras_steps.tolist():
        raise UserError('Missing values in est.respy.info')

    # We need to make sure that the size of the parameter vector does fit the initialization
    # file. For example, this might not be the case when the number of types is changed in the
    # initialization file and an update is requested with an earlier logfile.
    num_types, num_paras = len(init_dict['TYPE SHARES']['coeffs']) / 2 + 1, len(paras_steps)
    if num_paras != 53 + (num_types - 1) * 6:
        raise UserError('Info does not fit the current model specification')

    optim_paras = dist_optim_paras(paras_steps, True)
    shocks_coeffs = paras_steps[43:53]

    # Update initialization dictionary
    init_dict['COMMON']['coeffs'] = optim_paras['coeffs_common']
    init_dict['OCCUPATION A']['coeffs'] = optim_paras['coeffs_a']
    init_dict['OCCUPATION B']['coeffs'] = optim_paras['coeffs_b']
    init_dict['EDUCATION']['coeffs'] = optim_paras['coeffs_edu']
    init_dict['HOME']['coeffs'] = optim_paras['coeffs_home']
    init_dict['BASICS']['coeffs'] = optim_paras['delta']
    init_dict['SHOCKS']['coeffs'] = shocks_coeffs
    init_dict['TYPE SHARES']['coeffs'] = optim_paras['type_shares'][2:]
    init_dict['TYPE SHIFTS']['coeffs'] = optim_paras['type_shifts'].flatten()[4:]

    # We first print to an intermediate file as otherwise the original file is lost in case a
    # problem during printing occurs.
    write_init_file(init_dict, '.model.respy.ini')
    shutil.move('.model.respy.ini', init_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Update model initialization file with parameter '
                                                 'values from last step.')

    parser.add_argument('--init', action='store', dest='init_file', default='model.respy.ini',
                        help='initialization file')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run update
    scripts_update(args)
