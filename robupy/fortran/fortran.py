""" This module serves as the interface between the PYTHON code and the
FORTRAN implementations.
"""
# standard library
import os

import numpy as np

# project library
from robupy.fortran.fortran_auxiliary import write_robufort_initialization
from robupy.fortran.fortran_auxiliary import write_dataset
from robupy.fortran.fortran_auxiliary import get_results

from robupy.shared.constants import FORTRAN_DIR

''' Main function
'''

def fort_evaluate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, seed_emax, is_debug,
        min_idx, measure, edu_max, delta, level, num_draws_prob, num_agents,
        seed_prob, seed_data, request, data_frame):
    """ This function serves as the interface to the FORTRAN implementations.
    """
    # Antibugging
    if request == 'evaluate':
        assert data_frame is not None
    else:
        assert data_frame is None

    # Prepare ROBUFORT execution by collecting arguments and writing them to
    # the ROBUFORT initialization file.
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, seed_emax,
            is_debug, min_idx, measure, edu_max, delta, level,
            num_draws_prob, num_agents, seed_prob, seed_data, request)

    write_robufort_initialization(*args)

    # If an evaluation is requested, then a specially formatted dataset is
    # written to a scratch file. This eases the reading of the dataset in
    # FORTRAN.
    if request == 'evaluate':
        write_dataset(data_frame)

    # Call executable
    os.system('"' + FORTRAN_DIR + '/bin/robufort"')

    # Return arguments depends on the request.
    args = get_results(num_periods, min_idx)

    if request == 'solve':
        return args
    elif request == 'evaluate':
        # Get results and return
        crit_val = float(np.genfromtxt('.eval.robufort.dat'))
        os.unlink('.eval.robufort.dat')
        return crit_val, args
    else:
        raise NotImplementedError


def fort_solve(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, seed_emax, is_debug,
        measure, edu_max, min_idx, delta, level):
    """ This function serves as the interface to the FORTRAN implementations.
    """
    # Prepare ROBUFORT execution by collecting arguments and writing them to
    # the ROBUFORT initialization file.
    # TODO: ORder the arguments in write_robupfort so that I can do the
    # following args + (1, 1, 1,, solve).

    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, seed_emax,
            is_debug, min_idx, measure, edu_max, delta, level,
            1, 1, 1, 1, 'solve')

    write_robufort_initialization(*args)

    # Call executable
    os.system('"' + FORTRAN_DIR + '/bin/robufort"')

    # Return arguments depends on the request.
    args = get_results(num_periods, min_idx)

    return args