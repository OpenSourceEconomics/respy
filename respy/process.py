""" This module contains the interface to process a dataset from disk.
"""

# standard library
import pandas as pd
import numpy as np

# project library
from respy.python.process.process_auxiliary import check_process
from respy.python.shared.shared_auxiliary import get_robupy_obj
from respy.python.shared.shared_auxiliary import check_dataset


''' Main function
'''


def process(input):
    """ This function processes the dataset from disk.
    """
    # Process input
    robupy_obj = get_robupy_obj(input)

    # Antibugging
    assert robupy_obj.get_attr('is_locked')

    # Distribute class attributes
    num_agents_est = robupy_obj.get_attr('num_agents_est')

    num_periods = robupy_obj.get_attr('num_periods')

    file_est = robupy_obj.get_attr('file_est')

    is_debug = robupy_obj.get_attr('is_debug')

    # Construct auxiliary objects
    num_rows = num_agents_est * num_periods

    # Check integrity of processing request
    if is_debug:
        assert check_process(file_est + '.dat', robupy_obj)

    # Process dataset from files.
    data_frame = pd.read_csv(file_est + '.dat', delim_whitespace=True,
        header=-1, na_values='.', dtype={0: np.int, 1: np.int, 2: np.int,
        3: np.float, 4: np.int, 5: np.int, 6: np.int, 7: np.int},
        nrows=num_rows)

    # Check the dataset against the initialization files.
    check_dataset(data_frame, robupy_obj, 'est')

    # Finishing
    return data_frame

