"""Test functions to ensure the validity of data."""
import numpy as np


def check_estimation_data(df, optim_paras):
    """Check data for estimation.

    Parameters
    ----------
    optim_paras : dict
        Dictionary containing model optim_paras.
    df : pd.DataFrame
        Data for estimation.

    Raises
    ------
    AssertionError
        If data has not the expected format.

    """
    df = df.reset_index()

    n_periods = optim_paras["n_periods"]

    # 1. Identifier.
    # It is assumed in the likelihood function that Identifier starts at 0 and
    # increments in steps of one.
    unique = df["Identifier"].unique()
    assert (unique == np.arange(len(unique))).all()

    # 2. Period.
    assert df.Period.le(n_periods - 1).all()

    # 3. Choice.
    assert df.Choice.isin(optim_paras["choices"]).all()

    # 4. Wage.
    assert df.Wage.fillna(1).gt(0).all()

    # 8. Lagged_Choice.
    for i in range(1, optim_paras["n_lagged_choices"] + 1):
        assert df[f"Lagged_Choice_{i}"].isin(optim_paras["choices"]).all()

    if optim_paras["n_periods"] > 1 and optim_paras["n_lagged_choices"] > 0:
        choices = ["Choice"] + [
            f"Lagged_Choice_{i}" for i in range(1, optim_paras["n_lagged_choices"] + 1)
        ][:-1]

        for i in range(len(choices) - 1):
            lc = df.groupby("Identifier")[choices[i]].transform("shift").dropna()
            assert (
                df[choices[i + 1]].loc[~df.Period.le(i)].cat.codes == lc.cat.codes
            ).all()

    # Observable characteristics.
    for observable in optim_paras["observables"]:
        assert df[observable.title()].nunique() == len(
            optim_paras["observables"][observable]
        )

    # Others.
    assert df.drop(columns="Wage").notna().all().all()

    # We check individual state variables against the recorded choices.
    df.groupby("Identifier").apply(_check_state_variables, optim_paras=optim_paras)

    # Check that there are no duplicated observations for any period by agent.
    assert ~df.duplicated(subset=["Identifier", "Period"]).any()

    # Check that we observe the whole sequence of observations.
    max_periods_per_ind = df.groupby("Identifier").Period.max() + 1
    n_obs_per_ind = df.groupby("Identifier").size()
    assert (max_periods_per_ind == n_obs_per_ind).all()


def _check_state_variables(agent, optim_paras):
    """Check that state variables in the dataset.

    Construct the experience and schooling levels implied by the reported
    choices and compare them to the information provided in the dataset.

    """
    experiences = agent.iloc[0].filter(like="Experience_").to_numpy()

    for _, row in agent.iterrows():

        assert (experiences == row.filter(like="Experience_").to_numpy()).all()

        if row.Choice in optim_paras["choices_w_exp"]:
            index_of_choice = optim_paras["choices_w_exp"].index(row.Choice)
            experiences[index_of_choice] += 1


def check_simulated_data(optim_paras, df):
    """Check simulated data.

    This routine runs some consistency checks on the simulated dataset. Some more
    restrictions are imposed on the simulated dataset than the observed data.

    """
    df = df.copy()

    # Distribute class attributes
    n_periods = optim_paras["n_periods"]
    n_types = optim_paras["n_types"]

    # Run all tests available for the estimation data.
    check_estimation_data(df, optim_paras)

    # 9. Types.
    assert df.Type.max() <= n_types - 1
    assert df.Type.notna().all()
    assert df.groupby("Identifier").Type.nunique().eq(1).all()

    # Check that there are not missing wage observations if an agent is working. Also,
    # we check that if an agent is not working, there also is no wage observation.
    is_working = df["Choice"].isin(optim_paras["choices_w_wage"])
    assert df.Wage[is_working].notna().all()
    assert df.Wage[~is_working].isna().all()

    # Check that there are no missing observations and we follow an agent each period.
    df.groupby("Identifier").Period.nunique().eq(n_periods).all()
