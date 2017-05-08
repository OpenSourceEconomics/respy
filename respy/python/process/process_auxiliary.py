import numpy as np

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import DATA_LABELS_EST


def check_dataset_est(data_frame, respy_obj):
    """ This routine runs some consistency checks on the simulated data frame.
    """
    # Distribute class attributes
    num_periods, edu_spec = dist_class_attributes(respy_obj, 'num_periods', 'edu_spec')

    # Check that there are not missing values in any of the columns but for the wages information.
    for label in DATA_LABELS_EST:
        if label == 'Wage':
            continue
        assert ~ data_frame[label].isnull().any()

    # Checks for PERIODS. It can happen that the last period is deleted for all agents. Thus,
    # this is not a strict equality for observed data. It is for simulated data.
    dat = data_frame['Period']
    np.testing.assert_equal(dat.max() <= num_periods - 1, True)

    # Checks for CHOICE
    dat = data_frame['Choice'].isin([1, 2, 3, 4])
    np.testing.assert_equal(dat.all(), True)

    # Checks for WAGE
    dat = data_frame['Wage'].fillna(99) > 0.00
    np.testing.assert_equal(dat.all(), True)

    # Checks for EXPERIENCE. We also know that both need to take value of zero in the very first
    # period.
    for label in ['Experience_A', 'Experience_B']:
        dat = data_frame[label] >= 0.00
        np.testing.assert_equal(dat.all(), True)

        dat = data_frame[label][slice(None), 0] == 0
        np.testing.assert_equal(dat.all(), True)

    # Checks for LAGGED SCHOOLING. We also know that all individuals were in school when entering
    # the model.
    dat = data_frame['Lagged_Schooling'].isin([0, 1])
    np.testing.assert_equal(dat.all(), True)

    dat = data_frame['Lagged_Schooling'][slice(None), 0] == 1
    np.testing.assert_equal(dat.all(), True)

    # Checks for YEARS SCHOOLING. We also know that the initial years of schooling can only take
    # values specified in the initialization file.
    dat = data_frame['Years_Schooling'] >= 0.00
    np.testing.assert_equal(dat.all(), True)

    dat = data_frame['Years_Schooling'][slice(None), 0].isin(edu_spec['start'])
    np.testing.assert_equal(dat.all(), True)

    # Check that there are no duplicated observations for any period by agent.
    def check_unique_periods(group):
        np.testing.assert_equal(group['Period'].duplicated().any(), False)
    data_frame.groupby('Identifier').apply(check_unique_periods)

    # Check that we observe the whole sequence of observations and that they are in the right order.
    def check_series_observations(group):
        np.testing.assert_equal(group['Period'].tolist(), range(group['Period'].max() + 1))
    data_frame.groupby('Identifier').apply(check_series_observations)
