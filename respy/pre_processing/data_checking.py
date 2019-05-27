import numpy as np


def check_estimation_dataset(attr, df):
    """Run consistency checks on df."""
    # Distribute class attributes
    num_periods = attr["num_periods"]
    edu_spec = attr["edu_spec"]

    # Check that no variable but Wage contains missings.
    assert df.drop(columns="Wage").notna().all().all()

    # Checks for Period.
    assert df.Period.le(num_periods - 1).all()

    # Checks for Choice.
    assert df.Choice.isin([1, 2, 3, 4]).all()

    # Checks for Wage.
    assert df.Wage.fillna(1).gt(0).all()

    # Checks for Experience_*. We also know that both need to take value of zero in the
    # very first period.
    assert df.Experience_A.ge(0).all()
    assert df.Experience_B.ge(0).all()
    assert df.Experience_A.loc[df.Period.eq(0)].eq(0).all()
    assert df.Experience_B.loc[df.Period.eq(0)].eq(0).all()

    # We check individual state variables against the recorded choices.
    df.groupby("Identifier").apply(check_state_variables)

    # Checks for LAGGED ACTIVITY. Just to be sure, we also construct the correct lagged
    # activity here as well and compare it to the one provided in the dataset.
    assert df.Lagged_Choice.isin([1, 2, 3, 4]).all()
    assert df.Lagged_Choice.loc[df.Period.eq(0)].isin([3, 4]).all()

    lagged_choice = df.groupby("Identifier").Choice.transform("shift").dropna()
    assert (df.Lagged_Choice.loc[~df.Period.eq(0)] == lagged_choice).all()

    # Checks for Years_Schooling. We also know that the initial years of schooling can
    # only take values specified in the initialization file and no individual in our
    # estimation sample is allowed to have more than the maximum number of years of
    # education.
    assert df.Years_Schooling.ge(0).all()
    assert df.Years_Schooling.loc[df.Period.eq(0)].isin(edu_spec["start"]).all()
    assert df.Years_Schooling.le(edu_spec["max"]).all()

    # Check that there are no duplicated observations for any period by agent.
    def check_unique_periods(group):
        assert ~group["Period"].duplicated().any()

    df.groupby("Identifier").apply(check_unique_periods)

    # Check that we observe the whole sequence of observations and that they
    # are in the right order.
    def check_series_observations(group):
        np.testing.assert_equal(
            group["Period"].tolist(), list(range(group["Period"].max() + 1))
        )

    df.groupby("Identifier").apply(check_series_observations)


def check_state_variables(agent):
    """Check that state variables in the dataset.

    Construct the experience and schooling levels implied by the reported
    choices and compare them to the information provided in the dataset.

    """
    for _, row in agent.iterrows():
        period, choice = row.Period, row.Choice
        # We know that the level of experience is zero in the initial period
        # and we get the initial level of schooling.
        if period == 0:
            exp_a, exp_b, edu = 0.0, 0.0, row.Years_Schooling
        # Check statistics
        pairs = [
            (exp_a, "Experience_A"),
            (exp_b, "Experience_B"),
            (edu, "Years_Schooling"),
        ]
        for pair in pairs:
            stat, label = pair
            assert stat == row[label]
        # Update experience statistics.
        if choice == 1:
            exp_a += 1
        elif choice == 2:
            exp_b += 1
        elif choice == 3:
            edu += 1
        else:
            pass


def check_dataset_sim(data_frame, respy_obj):
    """ This routine runs some consistency checks on the simulated dataset.
    Some more restrictions are imposed on the simulated dataset than the
    observed data.

    """
    # TODO: Implement check for myopic agents where immediate rewards must be equal to
    #       total rewards.
    # Distribute class attributes
    num_agents = respy_obj.get_attr("num_agents_sim")
    num_periods = respy_obj.get_attr("num_periods")
    num_types = respy_obj.get_attr("num_types")

    # Some auxiliary functions for later
    def check_check_time_constant(group):
        np.testing.assert_equal(group["Type"].nunique(), 1)

    def check_number_periods(group):
        np.testing.assert_equal(group["Period"].count(), num_periods)

    # So, we run all checks on the observed dataset.
    check_estimation_dataset(data_frame, respy_obj)

    # Checks for PERIODS
    dat = data_frame["Period"]
    np.testing.assert_equal(dat.max(), num_periods - 1)

    # Checks for IDENTIFIER
    dat = data_frame["Identifier"]
    np.testing.assert_equal(dat.max(), num_agents - 1)

    # Checks for TYPES
    dat = data_frame["Type"]
    np.testing.assert_equal(dat.max() <= num_types - 1, True)
    np.testing.assert_equal(dat.isnull().any(), False)
    data_frame.groupby(level="Identifier").apply(check_check_time_constant)

    # Check that there are not missing wage observations if an agent is working. Also,
    # we check that if an agent is not working, there also is no wage observation.
    is_working = data_frame["Choice"].isin([1, 2])

    dat = data_frame["Wage"][is_working]
    np.testing.assert_equal(dat.isnull().any(), False)

    dat = data_frame["Wage"][~is_working]
    np.testing.assert_equal(dat.isnull().all(), True)

    # Check that there are no missing observations and we follow an agent each period.
    data_frame.groupby(level="Identifier").apply(check_number_periods)
