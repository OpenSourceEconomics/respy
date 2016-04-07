""" This module contains auxiliary functions for the estimation.
"""
# standard library
import numpy as np

# project library
from robupy.python.shared.shared_auxiliary import check_model_parameters
from robupy.python.shared.shared_auxiliary import check_dataset

''' Auxiliary functions
'''


def check_input(robupy_obj, data_frame):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert robupy_obj.get_attr('is_locked')

    if robupy_obj.get_attr('is_solved'):
        robupy_obj.reset()

    # Check that dataset aligns with model specification.
    check_dataset(data_frame, robupy_obj)

    # Finishing
    return True


def get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, shocks_cholesky, which, is_fixed, is_debug):
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

    # Select subset
    if which == 'free':
        x_free = []
        for i in range(26):
            if not is_fixed[i]:
                x_free += [x[i]]

        x = np.array(x_free)

    # Finishing
    return x


def get_model_parameters(x_all, is_debug):
    """ Update parameter values. The np.array type is maintained.
    """
    # Checks
    if is_debug:
        check_optimization_parameters(x_all)

    # Occupation A
    coeffs_a = x_all[0:6]

    # Occupation B
    coeffs_b = x_all[6:12]

    # Education
    coeffs_edu = x_all[12:15]

    # Home
    coeffs_home = x_all[15:16]

    # Cholesky
    shocks_cholesky = np.tile(0.0, (4, 4))

    shocks_cholesky[0:4, 0] = x_all[16:20]
    shocks_cholesky[1:4, 1] = x_all[20:23]
    shocks_cholesky[2:4, 2] = x_all[23:25]
    shocks_cholesky[3:4, 3] = x_all[25]

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


def process_block(list_, dict_, keyword):
    """ This function processes most parts of the initialization file.
    """
    # Distribute information
    name, val = list_[0], list_[1]

    # Prepare container.
    if (name not in dict_[keyword].keys()) and (name in ['coeff']):
        dict_[keyword][name] = []

    # Type conversion
    if name in ['m', 'maxfun']:
        val = int(val)
    elif name in ['file']:
        val = str(val)
    else:
        val = float(val)

    # Collect information
    dict_[keyword][name] = val

    # Finishing.
    return dict_


def process_cases(list_):
    """ Process cases and determine whether keyword or empty line.
    """
    # Antibugging
    assert (isinstance(list_, list))

    # Get information
    is_empty = (len(list_) == 0)

    if not is_empty:
        is_keyword = list_[0].isupper()
    else:
        is_keyword = False

    # Antibugging
    assert (is_keyword in [True, False])
    assert (is_empty in [True, False])

    # Finishing
    return is_empty, is_keyword

