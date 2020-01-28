"""Test the msm interface of respy."""
import pytest

from respy.interface import get_example_model
from respy.method_of_simulated_moments import get_diag_weighting_matrix
from respy.method_of_simulated_moments import get_msm_func
from respy.simulate import get_simulate_func
from respy.tests.utils import process_model_or_seed


@pytest.fixture
def inputs():
    calc_moments = {"Mean Wage": _calc_wage_mean, "Choices": _calc_choice_freq}

    params, options, df = get_example_model("kw_94_one", with_data=False)
    options["n_periods"] = 5
    simulate = get_simulate_func(params, options)
    df = simulate(params)

    empirical_moments = {
        "Choices": _replace_nans(_calc_choice_freq(df)),
        "Mean Wage": _replace_nans(_calc_wage_mean(df)),
    }

    weighting_matrix = get_diag_weighting_matrix(empirical_moments)

    return (
        params,
        options,
        calc_moments,
        _replace_nans,
        empirical_moments,
        weighting_matrix,
    )


def test_msm_zero(inputs):
    """ Test whether msm function successfully returns 0 for true parameter
    vector.
    """
    msm = get_msm_func(*inputs)
    msm_vector = get_msm_func(*inputs)

    assert msm(inputs[0]) == 0
    assert (msm_vector(inputs[0]) == 0).all()


def test_msm_nonzero(inputs):
    """ Test whether msm function successfully returns a value larger than 0
    for different deviations in the simulated set.
    """
    # 1. Different parameter vector.
    params = inputs[0].copy()
    params.loc["delta", "value"] = 0.8
    msm_params = get_msm_func(*inputs)

    # 2. Lower number of periods in the simulated dataset.
    msm_periods = get_msm_func(n_simulation_periods=4, *inputs)

    # 3. Different simulation seed for the simulated dataset.
    inputs[1]["simulation_seed"] = inputs[1]["simulation_seed"] + 100
    msm_seed = get_msm_func(*inputs)

    assert msm_params(params) > 0
    assert msm_periods(inputs[0]) > 0
    assert msm_seed(inputs[0]) > 0


@pytest.mark.parametrize("model_or_seed", ["kw_94_one", "kw_97_basic", None])
def test_randomness_msm(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)
    simulate = get_simulate_func(params, options)
    df = simulate(params)

    empirical_moments = _replace_nans(_calc_choice_freq(df))

    weighting_matrix = get_diag_weighting_matrix(empirical_moments)

    msm = get_msm_func(
        params,
        options,
        _calc_choice_freq,
        _replace_nans,
        empirical_moments,
        weighting_matrix,
    )

    assert msm(params) == 0


def _calc_choice_freq(df):
    return df.groupby("Period").Choice.value_counts(normalize=True).unstack()


def _calc_wage_mean(df):
    return df.groupby(["Period"])["Wage"].describe()["mean"]


def _replace_nans(df):
    return df.fillna(0)
