"""Test replication of key results in [1]_ and [2]_.

For [1]_, we test the following replications:

- Table 6: Only means and standard deviations for the exact solution.

For [2]_, we test the following replications:

- Tables 2.1-2.3: Choice probabilities per period.

References
----------
.. [1] Keane, M. P. and  Wolpin, K. I. (1994). `The Solution and Estimation of Discrete
       Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo
       Evidence <https://doi.org/10.2307/2109768>`__. *The Review of Economics and
       Statistics*, 76(4): 648-672.

.. [2] Keane, M. P. and  Wolpin, K. I. (1994b). `The Solution and Estimation of Discrete
       Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo
       Evidence <https://www.minneapolisfed.org/research/staff-reports/the-solution-and-
       estimation-of-discrete-choice-dynamic-programming-models-by-simulation-and-
       interpolation-monte-carlo-evidence>`_. *Federal Reserve Bank of Minneapolis*, No.
       181.

"""
import numpy as np
import pandas as pd
import pytest

import respy as rp
from respy.config import TEST_RESOURCES_DIR


pytestmark = pytest.mark.slow


@pytest.mark.end_to_end
@pytest.mark.precise
def test_table_6_exact_solution_row_mean_and_sd():
    """Replicate the first two rows of Table 6 in Keane and Wolpin (1994).

    In more detail, the mean effects and the standard deviations of a 500, 1000, and
    2000 dollar tuition subsidy on years of schooling and of experience in occupation a
    and occupation b based on 40 samples of 100 individuals using true parameters are
    tested.

    """
    # Specify the three different data sets.
    models = np.repeat(["one", "two", "three"], 2)
    tuition_subsidies = [0, 500, 0, 1000, 0, 2000]

    # Generate the 3 * 2 data sets as list of DataFrames by simulating with respective
    # tuition subsidy.
    data_frames = []
    for model, subsidy in zip(models, tuition_subsidies):
        params, options = rp.get_example_model(f"kw_94_{model}", with_data=False)
        options["simulation_agents"] = 4000
        simulate = rp.get_simulate_func(params, options)
        params.loc[("nonpec_edu", "at_least_twelve_exp_edu"), "value"] += subsidy
        data_frames.append(simulate(params))

    columns = ["Bootstrap_Sample", "Experience_Edu", "Experience_A", "Experience_B"]

    # Calculate the statistics based on 40 bootstrap samples á 100 individuals.
    bootstrapped_statistics = []
    for i, title in zip(range(0, 6, 2), ["kw_94_one", "kw_94_two", "kw_94_three"]):
        # Select sample with and without tuition subsidy.
        df_wo_ts = data_frames[i]
        df_w_ts = data_frames[i + 1]

        # Assign bootstrap sample number.
        df_wo_ts["Bootstrap_Sample"] = pd.cut(
            df_wo_ts.index.get_level_values(0), bins=40, labels=np.arange(1, 41)
        )
        df_w_ts["Bootstrap_Sample"] = pd.cut(
            df_w_ts.index.get_level_values(0), bins=40, labels=np.arange(1, 41)
        )

        # Calculate mean experiences.
        mean_exp_wo_ts = (
            df_wo_ts.query("Period == 39")[columns].groupby("Bootstrap_Sample").mean()
        )
        mean_exp_w_ts = (
            df_w_ts.query("Period == 39")[columns].groupby("Bootstrap_Sample").mean()
        )

        # Calculate bootstrap statistics.
        diff = (
            mean_exp_w_ts.subtract(mean_exp_wo_ts)
            .assign(Data=title)
            .reset_index()
            .set_index(["Data", "Bootstrap_Sample"])
            .stack()
            .unstack([0, 2])
        )
        bootstrapped_statistics.append(diff)

    rp_replication = pd.concat(
        [bs.agg(["mean", "std"]) for bs in bootstrapped_statistics], axis=1
    )

    # Expected values are taken from csv of table 6.
    kw_94_table_6 = pd.read_csv(
        TEST_RESOURCES_DIR / "kw_94_table_6.csv", index_col=0, header=[0, 1], nrows=2
    )

    # Test that standard deviations are very close.
    np.testing.assert_allclose(rp_replication.iloc[1], kw_94_table_6.iloc[1], atol=0.07)

    # Test that difference lies within one standard deviation.
    diff = rp_replication.iloc[0].to_numpy() - kw_94_table_6.iloc[0].to_numpy()
    assert (np.abs(diff) < kw_94_table_6.iloc[1]).all()


@pytest.mark.end_to_end
@pytest.mark.precise
@pytest.mark.parametrize(
    "model, table",
    zip(
        [f"kw_94_{model}" for model in ["one", "two", "three"]],
        [f"kw_94_wp_table_2_{i}.csv" for i in range(1, 4)],
    ),
)
def test_replication_of_choice_probabilities(model, table):
    """Replicate choice probabilities in Tables 2.1-2.3. in Keane and Wolpin (1994b).

    For each of the three parameterizations a data set is simulated and the choice
    probabilities for each period are compared to the numbers in the paper.

    """
    # Get choice probabilities from paper.
    expected = pd.read_csv(TEST_RESOURCES_DIR / table, index_col="period")

    # Simulate data for choice probabilities with more individuals to stabilize choice
    # probabilities. Also, more draws in the solution for better approximation of EMAX.
    params, options = rp.get_example_model(model, with_data=False)
    options["simulated_agents"] = 10_000

    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)

    result = (
        df.groupby("Period").Choice.value_counts(normalize=True).unstack().fillna(0)
    )

    np.testing.assert_allclose(expected, result, atol=0.1)
