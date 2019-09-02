"""Tests whether respy replicates results in Keane and Wolpin(1994)."""

import numpy as np
import pandas as pd
import pytest
import respy as rp


@pytest.mark.xfail
def test_table_6_exact_solution_row_mean_and_sd():
    """
    Tests whether respy replicates the Exact Solution row in table 6.
    In explicit, the mean effects and its standard deviation of a
    500, 1000, and 2000 dollar tuition fee on years of schooling and of
    experience in occupation 1 and occupation 2 based on 40 samples of
    100 persons using true paramter values are tested.
    
    """
    
    # Initialize the respective simulate function.
    params, options, _ = rp.get_example_model("kw_94_one")
    options["simulation_agents"] = 4000
    simulate = rp.get_simulate_func(params, options)
    
    # Specifiy the three different Data Sets.
    models = np.repeat(["one", "two", "three"], 2)
    tuition_subsidies = [0, 500, 0, 1000, 0, 2000]
    
    # Generate the 3*2 Data Sets as list of DataFrames by simulating with
    # respective tuition subsidy.
    data_frames = []
    for model, tuition_subsidy in zip(models, tuition_subsidies):
        params, _, _ = rp.get_example_model(f"kw_94_{model}")
        params.loc[("nonpec_edu", "hs_graduate"), "value"] += tuition_subsidy
        data_frames.append(simulate(params))
    
    # Calculate the statistics based on 40 bootstrap samples á 100 agents.
    bootstrapped_statistics = []
    for i, title in zip(range(0, 6, 2), ["Data Set One", "Data Set Two", "Data Set Three"]):
        df_wo_ts = data_frames[i]
        df_w_ts = data_frames[i + 1]
        df_wo_ts["Bootstrap_Sample"] = pd.cut(df_wo_ts.Identifier, bins=40, labels=np.arange(1, 41))
        df_w_ts["Bootstrap_Sample"] = pd.cut(df_w_ts.Identifier, bins=40, labels=np.arange(1, 41))
        mean_exp_wo_ts = df_wo_ts.loc[df_wo_ts.Period.eq(39), ["Bootstrap_Sample", "Experience_Edu", "Experience_A", "Experience_B"]].groupby("Bootstrap_Sample").mean()
        mean_exp_w_ts = df_w_ts.loc[df_w_ts.Period.eq(39), ["Bootstrap_Sample", "Experience_Edu", "Experience_A", "Experience_B"]].groupby("Bootstrap_Sample").mean()
        diff = mean_exp_w_ts - mean_exp_wo_ts
        diff["Data"] = title
        diff = diff.reset_index().set_index(["Data", "Bootstrap_Sample"]).stack().unstack([0, 2])
        bootstrapped_statistics.append(diff)
    
    rp_replication = pd.concat([bs.agg(["mean", "std"]) for bs in bootstrapped_statistics], axis=1)
    
    # Expected values are taken from csv in ..\resources.
    kw = pd.DataFrame(rp_replication, copy=True)
    kw_94_table_6 = pd.read_csv(r'resources/kw_94_table_6.csv', sep=";")
    kw.loc[['mean', 'std'],:] = kw_94_table_6.iloc[1:3,1:].astype(float).to_numpy()
    
    np.testing.assert_allclose(rp_replication, kw, rtol=0.02, atol=0)
