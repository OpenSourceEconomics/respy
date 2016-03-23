""" This module contains auxiliary functions for the estimation.
"""
# standard library
import numpy as np

# project library
from robupy.shared.auxiliary import check_model_parameters

''' Auxilary functions
'''


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