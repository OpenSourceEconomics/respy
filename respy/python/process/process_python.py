import pandas as pd
import numpy as np

from respy.python.process.process_auxiliary import check_process
from respy.python.shared.shared_auxiliary import check_dataset


def process(respy_obj):
    """ This function processes the dataset from disk.
    """
    # Antibugging
    assert respy_obj.get_attr('is_locked')

    # Distribute class attributes
    num_agents_est = respy_obj.get_attr('num_agents_est')

    num_periods = respy_obj.get_attr('num_periods')

    file_est = respy_obj.get_attr('file_est')

    is_debug = respy_obj.get_attr('is_debug')

    # Construct auxiliary objects
    num_rows = num_agents_est * num_periods

    # Check integrity of processing request
    if is_debug:
        assert check_process(file_est, respy_obj)

    # Process dataset from files.
    data_frame = pd.read_csv(file_est, delim_whitespace=True,
        header=-1, na_values='.', dtype={0: np.int, 1: np.int, 2: np.int,
        3: np.float, 4: np.int, 5: np.int, 6: np.int, 7: np.int},
        nrows=num_rows)

    # Check the dataset against the initialization files.
    check_dataset(data_frame, respy_obj, 'est')

    # Finishing
    return data_frame

