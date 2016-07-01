""" This module contains auxiliary functions for the estimation.
"""
# standard library
import numpy as np

# project library
from respy.python.shared.shared_auxiliary import check_optimization_parameters
from respy.python.shared.shared_auxiliary import check_model_parameters


def get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, which, paras_fixed, is_debug):
    """ Get optimization parameters.
    """
    # Checks
    if is_debug:
        args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)
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
    x[16:26] = shocks_cholesky[np.tril_indices(4)]

    # Checks
    if is_debug:
        check_optimization_parameters(x)

    # Select subset
    if which == 'free':
        x_free_curre = []
        for i in range(26):
            if not paras_fixed[i]:
                x_free_curre += [x[i]]

        x = np.array(x_free_curre)

    # Finishing
    return x


