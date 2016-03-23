""" This module contains auxiliary functions for the estimation.
"""
# standard library
import numpy as np

import os

# project library
from robupy.evaluate.evaluate_python import evaluate_python
from robupy.shared.auxiliary import check_model_parameters

''' Auxiliary functions
'''


def criterion(x, data_frame, edu_max, delta, edu_start, is_debug,
        is_interpolated, level, measure, min_idx, num_draws_emax, num_periods,
        num_points, is_ambiguous, periods_draws_emax, is_deterministic,
        is_myopic, num_agents, num_draws_prob, periods_draws_prob, is_python):
    """ This function provides the wrapper for optimization routines.
    """
    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        opt_get_model_parameters(x, is_debug)

    # Evaluate criterion function
    crit_val, _ = evaluate_python(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, shocks_cholesky, is_deterministic, is_interpolated,
        num_draws_emax, periods_draws_emax, is_ambiguous, num_periods,
        num_points, edu_start, is_myopic, is_debug, measure, edu_max, min_idx,
        delta, level, data_frame, num_agents, num_draws_prob,
        periods_draws_prob, is_python)

    if os.path.exists('.optimization.robupy.scratch'):
        logging_optimization('step', crit_val, x)

    print(crit_val)

    # Finishing
    return crit_val

def logging_optimization(which, *args):

    if which == 'start':

        crit_val, x0 = args

        info = np.array(np.array([0, crit_val]))

        np.savetxt(open('.optimization.robupy.scratch', 'wb'), info)

        with open('optimization.robupy.log', 'w') as out_file:
            out_file.write('STEP  {0:<5}'.format(0))
            out_file.write('VALUE {0:15.8f}\n\n'.format(crit_val))

        np.savetxt(open('.optimization.robupy.scratch', 'wb'), info)
        np.savetxt(open('start_paras.robupy.log', 'wb'), x0, fmt='%15.8f')


    elif which == 'step':

        crit_val, x = args

        info = np.genfromtxt('.optimization.robupy.scratch')
        current_num_step = int(info[0])
        current_crit_val = info[1]

        is_step = crit_val < current_crit_val

        if is_step:

            with open('optimization.robupy.log', 'a') as out_file:
                out_file.write('STEP  {0:<5}'.format(current_num_step + 1))
                out_file.write('VALUE {0:15.8f}\n\n'.format(crit_val))

            np.savetxt(open('step_paras.robupy.log', 'wb'), x, fmt='%15.8f')

            info = np.array(np.array([current_num_step + 1, crit_val]))
            np.savetxt(open('.optimization.robupy.scratch', 'wb'), info)

def opt_get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, shocks_cholesky, is_debug):
    """ Get optimization parameters.
    """
    # Checks
    if is_debug:
        args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
                shocks_cholesky]
        assert check_model_parameters(*args)

    # Initialize container
    x = np.tile(np.nan, 26)

    # Occupation A
    x[0:6] = coeffs_a

    # Occupation B
    x[6:12] = coeffs_b

    # Education
    x[12:15] = coeffs_edu

    # Home
    x[15:16] = coeffs_home

    # Shocks
    x[16:20] = shocks_cholesky[0:4, 0]
    x[20:23] = shocks_cholesky[1:4, 1]
    x[23:25] = shocks_cholesky[2:4, 2]
    x[25:26] = shocks_cholesky[3:4, 3]

    # Checks
    if is_debug:
        check_optimization_parameters(x)

    # Finishing
    return x


def opt_get_model_parameters(x, is_debug):
    """ Update parameter values. The np.array type is maintained.
    """
    # Checks
    if is_debug:
        check_optimization_parameters(x)

    # Occupation A
    coeffs_a = x[0:6]

    # Occupation B
    coeffs_b = x[6:12]

    # Education
    coeffs_edu = x[12:15]

    # Home
    coeffs_home = x[15:16]

    # Cholesky
    shocks_cholesky = np.tile(0.0, (4, 4))

    shocks_cholesky[0:4, 0] = x[16:20]
    shocks_cholesky[1:4, 1] = x[20:23]
    shocks_cholesky[2:4, 2] = x[23:25]
    shocks_cholesky[3:4, 3] = x[25]

    # Shocks
    shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)

    # Checks
    if is_debug:
        args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
               shocks_cholesky]
        assert check_model_parameters(*args)

    # Finishing
    return coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky


def check_optimization_parameters(x):
    """ Check optimization parameters.
    """
    # Perform checks
    assert (isinstance(x, np.ndarray))
    assert (x.dtype == np.float)
    assert (x.shape == (26,))
    assert (np.all(np.isfinite(x)))

    # Finishing
    return True

