"""Test the msm interface of respy."""
import copy

import pandas as pd
import pytest

from respy.interface import get_example_model
from respy.method_of_simulated_moments import get_diag_weighting_matrix
from respy.method_of_simulated_moments import get_moment_errors_func
from respy.simulate import get_simulate_func
from respy.tests.utils import process_model_or_seed


@pytest.fixture(scope="module")
def msm_args():
    calc_moments = {"Mean Wage": _calc_wage_mean, "Choices": _calc_choice_freq}

    params, options = get_example_model("kw_94_one", with_data=False)
    options["n_periods"] = 3
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


@pytest.mark.edge_case
@pytest.mark.end_to_end
@pytest.mark.parametrize("return_scalar", [True, False])
def test_msm_zero(msm_args, return_scalar):
    """Test whether MSM function successfully returns 0 for true parameter vector."""
    weighted_sum_squared_errors = get_moment_errors_func(
        *msm_args, return_scalar=return_scalar
    )

    if return_scalar:
        assert weighted_sum_squared_errors(msm_args[0]) == 0
    else:
        out = weighted_sum_squared_errors(msm_args[0])
        assert isinstance(out, pd.Series)
        assert (out == 0).all()


@pytest.mark.end_to_end
def test_msm_nonzero(msm_args):
    """MSM function successfully returns a value larger than 0 for different deviations
    in the simulated data set."""
    # 1. Different parameter vector.
    params = msm_args[0].copy()
    params.loc["delta", "value"] = 0.8
    msm_params = get_moment_errors_func(*msm_args)

    # 2. Lower number of periods in the simulated dataset.
    msm_periods = get_moment_errors_func(n_simulation_periods=4, *msm_args)

    # 3. Different simulation seed for the simulated dataset.
    options = copy.deepcopy(msm_args[1])
    options["simulation_seed"] = options["simulation_seed"] + 100
    msm_seed = get_moment_errors_func(msm_args[0], options, *msm_args[2:])

    assert msm_params(params) > 0
    assert msm_periods(params) > 0
    assert msm_seed(msm_args[0]) > 0


@pytest.mark.edge_case
@pytest.mark.end_to_end
@pytest.mark.parametrize("model_or_seed", ["kw_94_one", "kw_97_basic"])
def test_randomness_msm(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)
    simulate = get_simulate_func(params, options)
    df = simulate(params)

    empirical_moments = _replace_nans(_calc_choice_freq(df))

    weighting_matrix = get_diag_weighting_matrix(empirical_moments)

    weighted_sum_squared_errors = get_moment_errors_func(
        params,
        options,
        _calc_choice_freq,
        _replace_nans,
        empirical_moments,
        weighting_matrix,
    )

    assert weighted_sum_squared_errors(params) == 0


@pytest.mark.integration
def test_return_simulated_moments_for_msm(msm_args):
    """Return_simulated_moments."""
    weighted_sum_squared_errors = get_moment_errors_func(
        *msm_args, return_simulated_moments=True
    )
    fval, simulated_moments = weighted_sum_squared_errors(msm_args[0])

    assert isinstance(fval, float)
    assert isinstance(simulated_moments, (dict, list, pd.DataFrame, pd.Series))


@pytest.mark.integration
def test_return_comparison_plot_data_for_msm(msm_args):
    """Return_comparison_plot_data."""
    weighted_errors = get_moment_errors_func(
        *msm_args, return_scalar=False, return_comparison_plot_data=True
    )
    moment_errors, df = weighted_errors(msm_args[0])

    assert isinstance(moment_errors, pd.Series)
    assert isinstance(df, pd.DataFrame)


@pytest.mark.integration
def test_multiple_returns_msm(msm_args):
    """Raise error if moments and comparison plot data is requested."""
    with pytest.raises(ValueError, match="Can only return either"):
        get_moment_errors_func(
            *msm_args, return_simulated_moments=True, return_comparison_plot_data=True
        )


def _calc_choice_freq(df):
    return df.groupby("Period").Choice.value_counts(normalize=True).unstack()


def _calc_wage_mean(df):
    return df.groupby(["Period"])["Wage"].describe()["mean"]


def _replace_nans(df):
    return df.fillna(0)
