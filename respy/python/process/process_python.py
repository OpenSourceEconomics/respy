import pandas as pd

from respy.python.process.process_auxiliary import check_dataset_est
from respy.python.shared.shared_constants import FORMATS
from respy.python.shared.shared_constants import LABELS


def process(respy_obj):
    """ This function processes the dataset from disk.
    """
    # Distribute class attributes
    num_agents_est = respy_obj.get_attr('num_agents_est')
    file_est = respy_obj.get_attr('file_est')

    # Process dataset from files.
    data_frame = pd.read_csv(file_est, delim_whitespace=True, header=-1,
        na_values='.', dtype=FORMATS, names=LABELS)

    # We now subset the dataframe to include only the number of agents that
    # are requested for the estimation.
    data_frame.set_index(['Identifier'], drop=False, inplace=True)
    drop_indices = data_frame.index.unique()[num_agents_est:]
    data_frame.drop(drop_indices, inplace=True)

    data_frame.set_index(['Identifier', 'Period'], drop=False, inplace=True)

    # Check the dataset against the initialization files.
    check_dataset_est(data_frame, respy_obj)

    # Finishing
    return data_frame

