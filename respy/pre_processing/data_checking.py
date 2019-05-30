import numpy as np


def check_estimation_data(attr, df):
    """Check data for estimation.

    Parameters
    ----------
    attr : dict
        Dictionary containing model attributes.
    df : pd.DataFrame
        Data for estimation.

    Raises
    ------
    AssertionError
        If data has not the expected format.

    """
    df = df.copy()

    num_periods = attr["num_periods"]
    edu_spec = attr["edu_spec"]

    # 1. Identifier.

    # 2. Period.
    assert df.Period.le(num_periods - 1).all()

    # 3. Choice.
    assert df.Choice.isin([1, 2, 3, 4]).all()

    # 4. Wage.
    assert df.Wage.fillna(1).gt(0).all()

    # 5. Experience_A
    assert df.Experience_A.ge(0).all()
    assert df.Experience_A.loc[df.Period.eq(0)].eq(0).all()

    # 6. Experience_B
    assert df.Experience_B.ge(0).all()
    assert df.Experience_B.loc[df.Period.eq(0)].eq(0).all()

    # 7. Years_Schooling.
    assert df.Years_Schooling.ge(0).all()
    assert df.Years_Schooling.loc[df.Period.eq(0)].isin(edu_spec["start"]).all()
    assert df.Years_Schooling.le(edu_spec["max"]).all()

    # 8. Lagged_Choice.
    assert df.Lagged_Choice.isin([1, 2, 3, 4]).all()
    assert df.Lagged_Choice.loc[df.Period.eq(0)].isin([3, 4]).all()

    # Others.
    assert df.drop(columns="Wage").notna().all().all()

    # Compare Lagged_Choice and Choice for all but the first period.
    lagged_choice = df.groupby("Identifier").Choice.transform("shift").dropna()
    assert (df.Lagged_Choice.loc[~df.Period.eq(0)] == lagged_choice).all()

    # We check individual state variables against the recorded choices.
    df.groupby("Identifier").apply(_check_state_variables)

    # Check that there are no duplicated observations for any period by agent.
    assert ~df.duplicated(subset=["Identifier", "Period"]).any()

    # Check that we observe the whole sequence of observations.
    max_periods_per_ind = df.groupby("Identifier").Period.max() + 1
    num_obs_per_ind = df.groupby("Identifier").size()
    assert (max_periods_per_ind == num_obs_per_ind).all()


def _check_state_variables(agent):
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


def check_simulated_data(attr, optim_paras, df):
    """Check simulated data.

    This routine runs some consistency checks on the simulated dataset. Some more
    restrictions are imposed on the simulated dataset than the observed data.

    """
    df = df.copy()

    # Distribute class attributes
    num_periods = attr["num_periods"]
    num_types = optim_paras["num_types"]
    edu_max = attr["edu_spec"]["max"]

    # Run all tests available for the estimation data.
    check_estimation_data(attr, df)

    # 9. Types.
    assert df.Type.max() <= num_types - 1
    assert df.Type.notna().all()
    assert df.groupby("Identifier").Type.nunique().eq(1).all()

    # Check that there are not missing wage observations if an agent is working. Also,
    # we check that if an agent is not working, there also is no wage observation.
    is_working = df["Choice"].isin([1, 2])
    assert df.Wage[is_working].notna().all()
    assert df.Wage[~is_working].isna().all()

    # Check that there are no missing observations and we follow an agent each period.
    df.groupby("Identifier").Period.nunique().eq(num_periods).all()

    # If agents are myopic, we can test the equality of ex-post rewards and total
    # values.
    if df.Discount_Rate.eq(0).all():
        for choice in [1, 2]:
            is_working = df.Choice.eq(choice)

            ex_post_rew = f"Ex_Post_Reward_{choice}"
            general_rew = f"General_Reward_{choice}"

            df[ex_post_rew] = df.Wage + df[general_rew] + df.Common_Reward

            total_rewards = df[f"Total_Reward_{choice}"].loc[is_working]
            ex_post_reward = df[ex_post_rew].loc[is_working]

            np.testing.assert_array_almost_equal(total_rewards, ex_post_reward)

        for choice in [3, 4]:
            ex_post_rew = f"Ex_Post_Reward_{choice}"
            systematic_rew = f"Systematic_Reward_{choice}"
            shock_rew = f"Shock_Reward_{choice}"

            df[ex_post_rew] = df[systematic_rew] + df[shock_rew]

            # The equality does not hold if a state is inadmissible.
            cond = df.Years_Schooling.ne(edu_max)

            total_rewards = df[f"Total_Reward_{choice}"].loc[cond]
            ex_post_rewards = df[ex_post_rew].loc[cond]

            np.testing.assert_array_almost_equal(total_rewards, ex_post_rewards)
