""" This module contains some auxiliary functions for the evaluation of the
criterion function.
"""
# standard library

import numpy as np

# project library
from robupy.shared.auxiliary import check_dataset



def check_evaluation(str_, *args):
    """ Check integrity of criterion function.
    """
    if str_ == 'out':

        # Distribute input parameters
        crit_val, = args

        # Check quality
        assert isinstance(crit_val, float)
        assert np.isfinite(crit_val)

    elif str_ == 'in':

        # Distribute input parameters
        data_frame, robupy_obj, is_deterministic = args

        # Check quality
        check_dataset(data_frame, robupy_obj)

    # Finishing
    return True