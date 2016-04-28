""" This module contains auxiliary functions for the estimation.
"""
# standard library
import numpy as np

# project library
from respy.python.shared.shared_auxiliary import check_model_parameters
from respy.python.shared.shared_auxiliary import check_dataset


def check_input(respy_obj, data_frame):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert respy_obj.get_attr('is_locked')

    if respy_obj.get_attr('is_solved'):
        respy_obj.reset()

    # Check that dataset aligns with model specification.
    check_dataset(data_frame, respy_obj, 'est')

    # Finishing
    return True


def get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, which, paras_fixed, is_debug):
    """ Get optimization parameters.
    """
    # Construct Cholesky decomposition
    if np.count_nonzero(shocks_cov) == 0:
        shocks_cholesky = np.zeros((4, 4))
    else:
        shocks_cholesky = np.linalg.cholesky(shocks_cov)

    # Checks
    if is_debug:
        args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
                shocks_cholesky)
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
    x[16:26] = shocks_cholesky.T[np.triu_indices_from(shocks_cov)]

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


def dist_optim_paras(x_all_curre, is_debug):
    """ Update parameter values. The np.array type is maintained.
    """
    # Checks
    if is_debug:
        check_optimization_parameters(x_all_curre)

    # Occupation A
    coeffs_a = x_all_curre[0:6]

    # Occupation B
    coeffs_b = x_all_curre[6:12]

    # Education
    coeffs_edu = x_all_curre[12:15]

    # Home
    coeffs_home = x_all_curre[15:16]

    # Cholesky. The additional efforts to sort the values into the Cholesky
    # matrix are due to the attempt to keep the user interface as close to
    # the original paper. There the Cholesky matrix is presented as a flattened
    # upper triangular in Table 1. However, the authors switch to a
    # flattened lower triangular later in the paper.
    shocks_cholesky = np.tile(0.0, (4, 4))
    shocks_cholesky.T[0, 0:] = x_all_curre[16:20]
    shocks_cholesky.T[1, 1:] = x_all_curre[20:23]
    shocks_cholesky.T[2, 2:] = x_all_curre[23:25]
    shocks_cholesky.T[3, 3:] = x_all_curre[25:26]
    shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)

    # Checks
    if is_debug:
        args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
               shocks_cholesky)
        assert check_model_parameters(*args)

    # Collect arguments
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov)

    # Finishing
    return args


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

