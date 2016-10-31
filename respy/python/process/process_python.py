import pandas as pd
import numpy as np

from respy.python.simulate.simulate_auxiliary import check_dataset_sim


def process(respy_obj):
    """ This function processes the dataset from disk.
    """
    # Distribute class attributes
    num_agents_est = respy_obj.get_attr('num_agents_est')

    file_est = respy_obj.get_attr('file_est')

    # Process dataset from files.
    data_frame = pd.read_csv(file_est, delim_whitespace=True,
        header=-1, na_values='.', dtype={0: np.int, 1: np.int, 2: np.int,
        3: np.float, 4: np.int, 5: np.int, 6: np.int, 7: np.int})

    # We now subset the dataframe to include only the number of agents that
    # are requested for the estimation.
    data_frame.set_index([0], drop=False, inplace=True)
    drop_indices = data_frame.index.unique()[num_agents_est:]
    data_frame.drop(drop_indices, inplace=True)

    # Check the dataset against the initialization files.
    # TODO: Check for observed data ...
    #check_dataset(data_frame, respy_obj, 'est')

    # Finishing
    return data_frame

