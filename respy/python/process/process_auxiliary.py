import numpy as np

from respy.python.shared.shared_constants import LABELS


def check_dataset_est(data_frame, respy_obj):
    """ This routine runs some consistency checks on the simulated data frame.
    """
    # Distribute class attributes
    num_periods = respy_obj.get_attr('num_periods')

    # Check that there are not missing values in any of the columns but for
    # the wages information.
    for label in LABELS:
        if label == 'Wage':
            continue
        assert ~ data_frame[label].isnull().any()

    # Checks for PERIODS. It can happen that the last period is deleted for all
    # agents. Thus, this is not a strict equality for observed data. It is
    # for simulated data.
    dat = data_frame['Period']
    np.testing.assert_equal(dat.max() <= num_periods - 1, True)

    # Checks for CHOICE
    dat = data_frame['Choice'].isin([1, 2, 3, 4])
    np.testing.assert_equal(dat.all(), True)

    # Checks for WAGE
    dat = data_frame['Wage'].fillna(99) > 0.00
    np.testing.assert_equal(dat.all(), True)

    # Checks for EXPERIENCE
    for label in ['Experience A', 'Experience B']:
        dat = data_frame[label] >= 0.00
        np.testing.assert_equal(dat.all(), True)

    # Checks for LAGGED SCHOOLING
    dat = data_frame['Lagged Schooling'].isin([0, 1])
    np.testing.assert_equal(dat.all(), True)

    # Checks for YEARS SCHOOLING
    dat = data_frame['Years Schooling'] >= 0.00
    np.testing.assert_equal(dat.all(), True)

    # Check that there are no duplicated observations for any period by agent.
    def check_unique_periods(group):
        np.testing.assert_equal(group['Period'].duplicated().any(), False)

    data_frame.groupby('Identifier').apply(check_unique_periods)
