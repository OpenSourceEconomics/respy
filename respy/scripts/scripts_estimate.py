#!/usr/bin/env python

import numpy as np
import argparse
import os

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.estimate.estimate_auxiliary import get_optim_paras
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import get_est_info
from respy import estimate
from respy import RespyCls


def add_gradient_information(respy_obj):
    """ This function adds information about the gradient to the information
    files. It is not part of the estimation _modules as it breaks the design
    and requires to carry additional attributes. This results in considerable
    overhead, which appears justified at this point.
    """

    model_paras, is_debug, paras_fixed, derivatives = \
        dist_class_attributes(respy_obj, 'model_paras', 'is_debug',
            'paras_fixed', 'derivatives')

    # Auxiliary objects
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    # Construct starting values
    x_all_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, 'all', paras_fixed, is_debug)

    x_free_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, 'free', paras_fixed, is_debug)

    # Construct auxiliary information
    num_free = len(x_free_start)

    # The information about the gradient is simply added to the original
    # information later. Note that the original file is read before the
    # gradient evaluation. This is required as the information otherwise
    # accounts for the multiple function evaluation during the gradient
    # approximation scheme.
    original_lines = open('est.respy.info', 'r').readlines()
    fmt_ = '{0:<25}{1:>15}\n'
    original_lines[-5] = fmt_.format(*[' Number of Steps', 0])
    original_lines[-3] = fmt_.format(*[' Number of Evaluations', num_free])

    # Approximate gradient by forward finite differences.
    grad, ei = np.zeros((num_free,), float), np.zeros((27,), float)
    dfunc_eps = derivatives[1]

    # Making sure that the criterion is only evaluated at the relevant
    # starting values.
    respy_obj.unlock()
    respy_obj.set_attr('maxfun', 0)
    respy_obj.lock()

    _, f0 = estimate(respy_obj)

    for k, i in enumerate(np.where(np.logical_not(paras_fixed))[0].tolist()):
        x_baseline = x_all_start.copy()

        ei[i] = 1.0
        d = dfunc_eps * ei
        respy_obj.update_model_paras(x_baseline + d)

        _, f1 = estimate(respy_obj)

        grad[k] = (f1 - f0) / d[k]
        ei[i] = 0.0

    grad = np.random.uniform(0, 1, 27 - sum(paras_fixed)).tolist()
    norm = np.amax(np.abs(grad))

    # Write out extended information
    with open('est.respy.info', 'a') as out_file:
        # Insert information about gradient
        out_file.write('\n\n\n\n Gradient\n\n')
        fmt_ = '{0:>15}    {1:>15}\n\n'
        out_file.write(fmt_.format(*['Identifier', 'Start']))
        fmt_ = '{0:>15}    {1:15.4f}\n'

        # Iterate over all candidate values, but only write the free
        # ones to file. This ensure that the identifiers line up.
        for j in range(27):
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
        assert (os.path.exists('est.respy.info'))

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
        respy_obj.update_model_paras(get_est_info()['paras_step'])

    # Set maximum iteration count when only an evaluation of the criterion
    # function is requested.
    if single:
        respy_obj.unlock()
        respy_obj.set_attr('maxfun', 0)
        respy_obj.lock()

    # Optimize the criterion function.
    estimate(respy_obj)

    if gradient:
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
