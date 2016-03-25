""" This module contains some auxiliary functions for the evaluation of the
criterion function.
"""
# standard library

import numpy as np

# project library
from robupy.shared.auxiliary import check_dataset


def check_output(crit_val):
    """ Check integrity of criterion function.
    """
    assert (np.isfinite(crit_val))
    assert (isinstance(crit_val, float))

    # Finishing
    return True


def check_input(robupy_obj, data_frame):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert robupy_obj.get_attr('is_locked')

    # Check for previous solution.
    assert robupy_obj.get_attr('is_solved')

    # Check that dataset aligns with model specification.
    check_dataset(data_frame, robupy_obj)

    # Finishing
    return True
