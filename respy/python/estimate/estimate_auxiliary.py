import numpy as np

from respy.python.shared.shared_auxiliary import check_optimization_parameters
from respy.python.shared.shared_auxiliary import check_model_parameters
from respy.python.shared.shared_constants import MINISCULE_FLOAT


def get_optim_paras(level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, which, paras_fixed, is_debug):
    """ Get optimization parameters.
    """
    # Checks
    if is_debug:
        args = (level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                shocks_cholesky)
        assert check_model_parameters(*args)

    # Initialize container
    x = np.tile(np.nan, 27)

    # Level of Ambiguity
    x[0:1] = np.log(np.clip(level, MINISCULE_FLOAT, None))

    # Occupation A
    x[1:7] = coeffs_a

    # Occupation B
    x[7:13] = coeffs_b

    # Education
    x[13:16] = coeffs_edu

    # Home
    x[16:17] = coeffs_home

    # Shocks
    x[17:27] = shocks_cholesky[np.tril_indices(4)]

    # Checks
    if is_debug:
        check_optimization_parameters(x)

    # Select subset
    if which == 'free':
        x_free_curre = []
        for i in range(27):
            if not paras_fixed[i]:
                x_free_curre += [x[i]]

        x = np.array(x_free_curre)

    # Finishing
    return x


