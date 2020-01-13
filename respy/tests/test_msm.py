"""
Test the msm interface of respy.
"""
import pytest

from respy.interface import get_example_model
from respy.method_of_simulated_moments import get_diag_weighting_matrix
from respy.method_of_simulated_moments import get_msm_func


@pytest.fixture
def inputs():
    def calc_choice_freq(df):
        return df.groupby("Period").Choice.value_counts(normalize=True).unstack()

    def calc_wage_mean(df):
        return df.groupby(["Period"])["Wage"].describe()["mean"]

    def replace_nans(df):
        return df.fillna(0)

    calc_moments = {
        "Choices": calc_choice_freq,
        "Mean Wage": calc_wage_mean,
    }
    params, options, df_emp = get_example_model("kw_94_one")
    empirical_moments = {
        "Choices": calc_choice_freq(df_emp),
        "Mean Wage": calc_wage_mean(df_emp),
    }

    empirical_moments["Choices"] = replace_nans(empirical_moments["Choices"])
    empirical_moments["Mean Wage"] = replace_nans(empirical_moments["Mean Wage"])

    weighting_matrix = get_diag_weighting_matrix(empirical_moments)

    return (
        params,
        options,
        calc_moments,
        replace_nans,
        empirical_moments,
        weighting_matrix,
    )


def test_msm_zero(inputs):
    """ Test whether msm function successfully returns 0 for true parameter
    vector.
    """
    msm = get_msm_func(return_scalar=True, *inputs)
    msm_vector = get_msm_func(return_scalar=False, *inputs)

    scalar = msm(inputs[0])
    vector = msm_vector(inputs[0])

    assert scalar == 0 and (vector == 0).all()


def test_msm_seed(inputs):
    """ Test whether msm function successfully returns a value larger than 0
    for a different simulation seed.
    """
    inputs[1]["simulation_seed"] = inputs[1]["simulation_seed"] + 100
    msm = get_msm_func(return_scalar=True, *inputs)
    scalar = msm(inputs[0])

    assert scalar > 0
