import pandas as pd
import numpy as np

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.process.process_auxiliary import check_dataset_est
from respy.python.shared.shared_constants import DATA_FORMATS_EST
from respy.python.shared.shared_constants import DATA_LABELS_EST


def process(respy_obj):
    """ This function processes the dataset from disk.
    """
    # Distribute class attributes
    num_agents_est, file_est, edu_spec = dist_class_attributes(respy_obj, 'num_agents_est',
        'file_est', 'edu_spec')

    # Process dataset from files.
    data_frame = pd.read_csv(file_est, delim_whitespace=True, header=0, na_values='.')
    data_frame.set_index(['Identifier', 'Period'], drop=False, inplace=True)

    # We only keep the information that is relevant for the estimation. Once that is done,
    # we can also impose some type restrictions.
    data_frame = data_frame[DATA_LABELS_EST]
    data_frame = data_frame.astype(DATA_FORMATS_EST)

    # We want to restrict the sample to meet the specified initial conditions.
    cond = data_frame['Years_Schooling'].loc[:, 0].isin(edu_spec['start'])
    data_frame.set_index(['Identifier'], drop=False, inplace=True)
    data_frame = data_frame.loc[cond]

    # We now subset the dataframe to include only the number of agents that are requested for the
    # estimation.
    drop_indices = data_frame.index.unique()[num_agents_est:]
    data_frame.drop(drop_indices, inplace=True)

    data_frame.set_index(['Identifier', 'Period'], drop=False, inplace=True)

    # We need to make sure that we only use less or equal to the number of individuals requested
    # for the estimation. It can be less than the number requested if we drop some initial
    # conditions.
    dat = len(data_frame['Identifier'].unique())
    np.testing.assert_equal(0 < dat <= num_agents_est, True)

    # Check the dataset against the initialization files.
    check_dataset_est(data_frame, respy_obj)

    # Finishing
    return data_frame

