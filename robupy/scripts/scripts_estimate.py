#!/usr/bin/env python
""" This script serves as a command line tool to ease the estimation of the
model.
"""

# standard library
from scipy.optimize import approx_fprime

import numpy as np

import argparse
import os

# project library
from robupy.python.estimate.estimate_auxiliary import dist_optim_paras
from robupy.python.estimate.estimate_auxiliary import get_optim_paras

from robupy.python.shared.shared_auxiliary import dist_class_attributes
from robupy.python.shared.shared_auxiliary import dist_model_paras
from robupy.python.shared.shared_auxiliary import create_draws

from robupy.python.estimate.estimate_wrapper import OptimizationClass

from robupy.process import process
from robupy.read import read
from robupy import estimate

""" Auxiliary function
"""


def add_gradient_information(robupy_obj, data_frame):
    """ This function adds information about the gradient to the information
    files. It is not part of the estimation modules as it breaks the design
    and requires to carry additional attributes. This results in considerable
    overhead, which appears justified at this point.
    """
    data_array = data_frame.as_matrix()

    model_paras, num_periods, num_agents_est, edu_start, seed_sim, \
        is_debug, file_sim, edu_max, delta, num_draws_prob, seed_prob, \
        num_draws_emax, seed_emax, level, min_idx, is_ambiguous, \
        is_deterministic, is_myopic, is_interpolated, num_points, version, \
        maxiter, optimizer, paras_fixed, tau, file_opt = \
        dist_class_attributes(robupy_obj,
            'model_paras', 'num_periods', 'num_agents_est', 'edu_start',
            'seed_sim', 'is_debug', 'file_sim', 'edu_max', 'delta',
            'num_draws_prob', 'seed_prob', 'num_draws_emax', 'seed_emax',
            'level',  'min_idx', 'is_ambiguous',
            'is_deterministic', 'is_myopic', 'is_interpolated',
            'num_points', 'version', 'maxiter', 'optimizer', 'paras_fixed',
            'tau', 'file_opt')

    # Auxiliary objects
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    # Construct starting values
    x_all_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cov, 'all', paras_fixed, is_debug)

    x_free_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cov, 'free', paras_fixed, is_debug)

    # Draw standard normal deviates for the solution and evaluation step.
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug)

    # Collect arguments for the evaluation of the criterion function.
    args = (is_deterministic, is_interpolated, num_draws_emax,is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug,
        edu_max, min_idx, delta, level, data_array, num_agents_est,
        num_draws_prob, tau, periods_draws_emax, periods_draws_prob)

    # Prepare evaluation of the criterion function.
    opt_obj = OptimizationClass()

    opt_obj.set_attr('args', args)

    opt_obj.set_attr('optimizer', optimizer)

    opt_obj.set_attr('x_info', (x_all_start, paras_fixed))

    opt_obj.set_attr('file_opt', file_opt)

    opt_obj.set_attr('version', version)

    opt_obj.set_attr('maxiter', 0)

    opt_obj.lock()

    # The information about the gradient is simply added to the original
    # information later. Note that the original file is read before the
    # gradient evaluation. This is required as the information otherwise
    # accounts for the multiple function evaluation during the gradient
    # approximation scheme.
    original_lines = open('optimization.robupy.info', 'r').readlines()
    fmt_ = '{0:<25}{1:>15}\n'
    original_lines[-5] = fmt_.format(*[' Number of Steps', 0])
    original_lines[-3] = fmt_.format(*[' Number of Evaluations', len(x_free_start)])

    # Approximate gradient by forward finite differences.
    epsilon = 1.4901161193847656e-08
    grad = approx_fprime(x_free_start, opt_obj.crit_func, epsilon, *args).tolist()
    norm = np.amax(np.abs(grad))
    # Write out extended information
    with open('optimization.robupy.info', 'w') as out_file:
        for i, line in enumerate(original_lines):
            out_file.write(line)
            # Insert information about gradient
            if i == 6:
                out_file.write('\n Gradient\n\n')
                fmt_ = '{0:>15}    {1:>15}\n\n'
                out_file.write(fmt_.format(*['Identifier', 'Start']))
                fmt_ = '{0:>15}    {1:15.4f}\n'

                # Iterate over all candidate values, but only write the free
                # ones to file. This ensure that the identifiers line up.
                for j in range(26):

                    if j < 16:
                        is_fixed = paras_fixed[j]
                    else:
                        is_fixed = paras_fixed[16]

                    if not is_fixed:
                        values = [j, grad.pop(0)]
                        out_file.write(fmt_.format(*values))

                out_file.write('\n')

                # Add value of infinity norm
                values = ['Norm', norm]
                out_file.write(fmt_.format(*values))
                out_file.write('\n\n')


def dist_input_arguments(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    init_file = args.init_file
    gradient = args.gradient
    resume = args.resume
    single = args.single

    # Check attributes
    assert (single in [True, False])
    assert (resume in [False, True])
    assert (os.path.exists(init_file))

    if gradient:
        # The gradient information is only provided if a single function
        # evaluation is requested.
        assert single

    if resume:
        assert (os.path.exists('paras_steps.robupy.log'))

    # Finishing
    return resume, single, init_file, gradient


""" Main function
"""


def scripts_estimate(resume, single, init_file, gradient):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification.
    robupy_obj = read(init_file)

    # Update parametrization of the model if resuming from a previous
    # estimation run.
    if resume:
        x0 = np.genfromtxt('paras_steps.robupy.log')
        args = dist_optim_paras(x0, True)
        robupy_obj.update_model_paras(*args)

    # Set maximum iteration count when only an evaluation of the criterion
    # function is requested.
    if single:
        robupy_obj.unlock()
        robupy_obj.set_attr('maxiter', 0)
        robupy_obj.lock()

    # Process dataset
    data_frame = process(robupy_obj)

    # Optimize the criterion function.
    estimate(robupy_obj, data_frame)

    if gradient:
        add_gradient_information(robupy_obj, data_frame)

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description =
        'Start of estimation run with the ROBUPY package.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--resume', action ='store_true',  dest='resume',
        default=False, help='resume estimation run')

    parser.add_argument('--single', action='store_true', dest='single',
        default=False, help='single evaluation')

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.robupy.ini', help='initialization file')

    parser.add_argument('--gradient', action='store_true', dest='gradient',
        default=False, help='gradient information')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run estimation 
    scripts_estimate(*args)
