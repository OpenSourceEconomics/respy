""" This module provides the interface to the functionality needed to
evaluate the likelihood function.
"""

# standard library
import numpy as np
import os

# project library
from robupy.fortran.auxiliary import _write_robufort_initialization
from robupy.fortran.auxiliary import _add_results
from robupy.constants import HUGE_FLOAT

# module-wide variables
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

''' Main function
'''


def evaluate_fortran(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, seed_emax,
            is_debug, min_idx, measure, edu_max, delta, level,
            num_draws_prob, num_agents, seed_prob, seed_data, request,
                     data_frame):
    """ Solve dynamic programming using FORTRAN.
    """
    # Prepare ROBUFORT execution
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, seed_emax,
            is_debug, min_idx, measure, edu_max, delta, level,
            num_draws_prob, num_agents, seed_prob, seed_data, 'evaluate')

    _write_robufort_initialization(*args)

    _write_dataset(data_frame)

    # Call executable
    os.system('"' + PACKAGE_PATH + '/bin/robufort"')

    # Add results
    eval_ = None
    if request == 'evaluate':
        eval_ = float(np.genfromtxt('.eval.robufort.dat'))
        os.unlink('.eval.robufort.dat')


    # Finishing
    return eval_

''' Auxiliary function
'''


def _write_dataset(data_frame):
    """ Write the dataset to a temporary file. Missing values are set
    to large values.
    """
    with open('.data.robufort.dat', 'w') as file_:
        data_frame.to_string(file_, index=False,
            header=None, na_rep=str(HUGE_FLOAT))

    # An empty line is added as otherwise this might lead to problems on the
    # TRAVIS servers. The FORTRAN routine read_dataset() raises an error.
    with open('.data.robufort.dat', 'a') as file_:
        file_.write('\n')


















