""" This module allows to process a dataset from disk.
"""

# standard library
import pandas as pd
import numpy as np

import os

# project library
from robupy.auxiliary import check_dataset

''' Main function
'''


def process(data_file, robupy_obj):
    """ This function processes the dataset from disk.
    """

    # Antibugging
    assert _check_process(data_file, robupy_obj)

    # Process dataset from files.
    data_frame = pd.read_csv(data_file, delim_whitespace=True, header=-1,
        na_values='.', dtype={0: np.int, 1: np.int, 2: np.int, 3: np.float,
        4: np.int, 5: np.int, 6: np.int, 7: np.int})

    # Check the dataset against the initialization files.
    check_dataset(data_frame, robupy_obj)

    # Finishing
    return data_frame

''' Auxiliary functions
'''


def _check_process(data_file, robupy_obj):
    """ Check likelihood calculation.
    """

    assert (os.path.exists(data_file))
    assert (robupy_obj.get_status())

    # Finishing
    return True