#!/usr/bin/env python

# standard library
import argparse
import os

import numpy as np
from scipy.optimize import approx_fprime

# project library
from respy.python.estimate.estimate_auxiliary import get_optim_paras

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import create_draws

from respy.python.estimate.estimate_wrapper import OptimizationClass

from respy.python.process.process_python import process

from respy import estimate
from respy import RespyCls


def add_gradient_information(respy_obj):
    """ This function adds information about the gradient to the information
    files. It is not part of the estimation modules as it breaks the design
    and requires to carry additional attributes. This results in considerable
    overhead, which appears justified at this point.
    """
    data_array = process(respy_obj).as_matrix()

    model_paras, num_periods, num_agents_est, edu_start, is_debug, edu_max, \
        delta, num_draws_prob, seed_prob, num_draws_emax, seed_emax, \
        min_idx, is_myopic, is_interpolated, num_points_interp, version, \
        optimizer_used, paras_fixed, tau, optimizer_options = \
            dist_class_attributes(respy_obj,
                'model_paras', 'num_periods', 'num_agents_est', 'edu_start',
                'is_debug', 'edu_max', 'delta', 'num_draws_prob', 'seed_prob',
                'num_draws_emax', 'seed_emax', 'min_idx', 'is_myopic',
                'is_interpolated', 'num_points_interp', 'version', 'optimizer_used',
                'paras_fixed', 'tau', 'optimizer_options')

    # Auxiliary objects
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    # Construct starting values
    x_all_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, 'all', paras_fixed, is_debug)

    x_free_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, 'free', paras_fixed, is_debug)

    # Draw standard normal deviates for the solution and evaluation step.
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug)

    # Collect arguments for the evaluation of the criterion function.
    args = (is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_start, is_debug, edu_max, min_idx, delta, data_array,
        num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob)

    # Prepare evaluation of the criterion function.
    opt_obj = OptimizationClass()

    opt_obj.set_attr('args', args)

    opt_obj.set_attr('optimizer_options', optimizer_options)

    opt_obj.set_attr('x_info', (x_all_start, paras_fixed))

    opt_obj.set_attr('optimizer_used', optimizer_used)

    opt_obj.set_attr('version', version)

    opt_obj.set_attr('maxiter', 0)

    opt_obj.lock()

    # The information about the gradient is simply added to the original
    # information later. Note that the original file is read before the
    # gradient evaluation. This is required as the information otherwise
    # accounts for the multiple function evaluation during the gradient
    # approximation scheme.
    original_lines = open('optimization.respy.info', 'r').readlines()
    fmt_ = '{0:<25}{1:>15}\n'
    original_lines[-5] = fmt_.format(*[' Number of Steps', 0])
    original_lines[-3] = fmt_.format(*[' Number of Evaluations', len(x_free_start)])

    # Approximate gradient by forward finite differences.
    epsilon = optimizer_options['SCIPY-BFGS']['epsilon']
    grad = approx_fprime(x_free_start, opt_obj.crit_func, epsilon, *args).tolist()
    norm = np.amax(np.abs(grad))
    # Write out extended information
    with open('optimization.respy.info', 'w') as out_file:
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
                    is_fixed = paras_fixed[j]
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
        assert (os.path.exists('paras_steps.respy.log'))

    # Finishing
    return resume, single, init_file, gradient


def scripts_estimate(resume, single, init_file, gradient):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification.
    respy_obj = RespyCls(init_file)

    # Update parametrization of the model if resuming from a previous
    # estimation run.
    if resume:
        x0 = np.genfromtxt('paras_steps.respy.log')
        respy_obj.update_model_paras(x0)

    # Set maximum iteration count when only an evaluation of the criterion
    # function is requested.
    if single:
        respy_obj.unlock()
        respy_obj.set_attr('maxiter', 0)
        respy_obj.lock()

    # Optimize the criterion function.
    estimate(respy_obj)

    if gradient:
        # TODO: This is only running with PYTHON
        add_gradient_information(respy_obj)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Start of estimation run with the RESPY package.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--resume', action='store_true', dest='resume',
        default=False, help='resume estimation run')

    parser.add_argument('--single', action='store_true', dest='single',
        default=False, help='single evaluation')

    parser.add_argument('--init_file', action='store', dest='init_file',
        default='model.respy.ini', help='initialization file')

    parser.add_argument('--gradient', action='store_true', dest='gradient',
        default=False, help='gradient information')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run estimation
    scripts_estimate(*args)
