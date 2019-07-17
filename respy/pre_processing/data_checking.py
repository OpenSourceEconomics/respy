import numpy as np


def check_estimation_data(df, options):
    """Check data for estimation.

    Parameters
    ----------
    options : dict
        Dictionary containing model options.
    df : pd.DataFrame
        Data for estimation.

    Raises
    ------
    AssertionError
        If data has not the expected format.

    """
    df = df.copy().reset_index()
    choices = options["choices"]

    n_periods = options["n_periods"]

    # 1. Identifier.

    # 2. Period.
    assert df.Period.le(n_periods - 1).all()

    # 3. Choice.
    assert df.Choice.isin(options["choices"]).all()

    # 4. Wage.
    assert df.Wage.fillna(1).gt(0).all()

    for choice in options["choices_w_exp"]:
        assert (
            df[f"Experience_{choice.title()}"].ge(choices[choice]["start"].min()).all()
        )
        assert df[f"Experience_{choice.title()}"].le(choices[choice]["max"]).all()

    # 8. Lagged_Choice.
    assert df.Lagged_Choice.isin(options["choices"]).all()

    # Others.
    assert df.drop(columns="Wage").notna().all().all()

    if options["n_periods"] > 1:
        # Compare Lagged_Choice and Choice for all but the first period.
        lagged_choice = df.groupby("Identifier").Choice.transform("shift").dropna()
        assert (
            df.Lagged_Choice.loc[~df.Period.eq(0)].cat.codes == lagged_choice.cat.codes
        ).all()

    # We check individual state variables against the recorded choices.
    df.groupby("Identifier").apply(_check_state_variables)

    # Check that there are no duplicated observations for any period by agent.
    assert ~df.duplicated(subset=["Identifier", "Period"]).any()

    # Check that we observe the whole sequence of observations.
    max_periods_per_ind = df.groupby("Identifier").Period.max() + 1
    n_obs_per_ind = df.groupby("Identifier").size()
    assert (max_periods_per_ind == n_obs_per_ind).all()


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
            exp_a, exp_b, edu = 0.0, 0.0, row.Experience_Edu
        else:
            pass
        # Check statistics
        pairs = [
            (exp_a, "Experience_A"),
            (exp_b, "Experience_B"),
            (edu, "Experience_Edu"),
        ]
        for pair in pairs:
            stat, label = pair
            assert stat == row[label]
        # Update experience statistics.
        if choice == "a":
            exp_a += 1
        elif choice == "b":
            exp_b += 1
        elif choice == "edu":
            edu += 1
        else:
            pass


def check_simulated_data(options, df):
    """Check simulated data.

    This routine runs some consistency checks on the simulated dataset. Some more
    restrictions are imposed on the simulated dataset than the observed data.

    """
    df = df.copy()

    # Distribute class attributes
    n_periods = options["n_periods"]
    n_types = options["n_types"]
    edu_max = options["choices"]["edu"]["max"]  # noqa: F841

    # Run all tests available for the estimation data.
    check_estimation_data(df, options)

    # 9. Types.
    assert df.Type.max() <= n_types - 1
    assert df.Type.notna().all()
    assert df.groupby("Identifier").Type.nunique().eq(1).all()

    # Check that there are not missing wage observations if an agent is working. Also,
    # we check that if an agent is not working, there also is no wage observation.
    is_working = df["Choice"].isin(options["choices_w_wage"])
    assert df.Wage[is_working].notna().all()
    assert df.Wage[~is_working].isna().all()

    # Check that there are no missing observations and we follow an agent each period.
    df.groupby("Identifier").Period.nunique().eq(n_periods).all()

    # If agents are myopic, we can test the equality of ex-post rewards and total
    # values.
    if df.Discount_Rate.eq(0).all():
        for choice in options["choices"]:
            if choice in options["choices_w_wage"]:
                fu_lab = f"Flow_Utility_{choice}"
                nonpec_lab = f"Nonpecuniary_Reward_{choice}"

                df[fu_lab] = df.Wage + df[nonpec_lab]

                is_working = "Choice == @choice"
                value_function = df[f"Value_Function_{choice}"].query(is_working)
                flow_utility = df[fu_lab].query(is_working)

                np.testing.assert_array_almost_equal(value_function, flow_utility)

            else:
                fu_lab = f"Flow_Utility_{choice}"
                nonpec_lab = f"Nonpecuniary_Reward_{choice}"
                shock_rew = f"Shock_Reward_{choice}"

                df[fu_lab] = df[nonpec_lab] + df[shock_rew]

                # The equality does not hold if a state is inadmissible.
                not_max_education = "Years_Schooling != @edu_max"
                value_function = df[f"Value_Function_{choice}"].query(not_max_education)
                flow_utility = df[fu_lab].query(not_max_education)

                np.testing.assert_array_almost_equal(value_function, flow_utility)
