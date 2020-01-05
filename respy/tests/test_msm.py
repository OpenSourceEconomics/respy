"""
Test the msm interface of respy!
"""
import numpy as np
import pytest

import respy as rp
from respy.method_of_simulated_moments import get_diag_weighting_matrix
from respy.method_of_simulated_moments import msm


@pytest.fixture
def inputs():
    def calc_choice_freq(df):
        return df.groupby("Period").Choice.value_counts(normalize=True).unstack()

    def calc_wage_distr(df):
        return df.groupby(["Period"])["Wage"].describe()[["mean", "std"]]

    def fill_nans_zero(df):
        return df.fillna(0)

    calc_moments = [calc_choice_freq, calc_wage_distr]
    replace_nans = [fill_nans_zero, fill_nans_zero]

    params, options, df_emp = rp.get_example_model("kw_94_one")
    empirical_moments = [calc_choice_freq(df_emp), calc_wage_distr(df_emp)]
    empirical_moments[0] = fill_nans_zero(empirical_moments[0])
    empirical_moments[1] = fill_nans_zero(empirical_moments[1])
    weighting_matrix = get_diag_weighting_matrix(empirical_moments)

    return params, options, calc_moments, replace_nans, empirical_moments, weighting_matrix
 
def test_msm_base(inputs):
    """ Test whether msm function successfully returns 0 for true parameter 
    vector.
    """
    rslt = msm(
        params=inputs[0],
        options=inputs[1],
        calc_moments=inputs[2],
        replace_nans=inputs[3],
        empirical_moments=inputs[4],
        weighting_matrix=inputs[5],
        return_scalar=True,
    )
    assert rslt == 0
