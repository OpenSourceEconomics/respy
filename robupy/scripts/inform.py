#!/usr/bin/env python
""" This script provides useful information during an estimation run.
"""

# standard library
import numpy as np

import argparse

# project library
from robupy.estimate.estimate_auxiliary import opt_get_model_parameters

""" Auxiliary function
"""


def distribute_input_arguments(parser):
    """ Check input for script.
    """
    # Parse arguments
    parser.parse_args()


""" Main function
"""


def inform():
    """ Provide some additional information during estimation run.
    """
    # Read in the information from the optimization run.
    paras_curre = np.genfromtxt('paras_curre.robupy.log')
    paras_start = np.genfromtxt('paras_start.robupy.log')
    paras_steps = np.genfromtxt('paras_steps.robupy.log')

    # Write information to file.
    with open('optimization.robupy.info', 'w') as out_file:
        # Write out information about the optimization parameters directly.
        out_file.write('\n Optimization Parameters\n\n')
        fmt_ = '{0:>15}    {1:>15}    {2:>15}    {3:>15}\n\n'
        out_file.write(fmt_.format(*['Identifier', 'Start', 'Step', 'Current']))
        fmt_ = '{0:>15}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n'
        for i in range(len(paras_curre)):
            paras = [i, paras_start[i], paras_steps[i], paras_curre[i]]
            out_file.write(fmt_.format(*paras))

        # Write out the current covariance matrix of the reward shocks.
        out_file.write('\n\n Current Covariance Matrix \n\n')
        shocks_cov = opt_get_model_parameters(paras_curre, True)[4]
        fmt_ = '{0:15.4f}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n'
        for i in range(4):
            out_file.write(fmt_.format(*shocks_cov[i, :]))


''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description =
        'Get additional information during estimation run.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    distribute_input_arguments(parser)

    inform()

