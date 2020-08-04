import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from scipy import special

from respy.likelihood import _logsumexp
from respy.likelihood import get_log_like_func
from respy.simulate import get_simulate_func
from respy.tests.utils import process_model_or_seed


@pytest.mark.integration
@pytest.mark.parametrize("model", ["robinson_crusoe_basic"])
def test_return_comparison_plot_data_for_likelihood(model):
    params, options = process_model_or_seed(model)
    options["n_periods"] = 3

    simulate = get_simulate_func(params, options)
    df = simulate(params)

    log_like = get_log_like_func(params, options, df, return_comparison_plot_data=False)
    log_like = log_like(params)

    assert isinstance(log_like, float)

    log_like = get_log_like_func(params, options, df, return_comparison_plot_data=True)
    log_like, df = log_like(params)

    assert isinstance(log_like, float)
    assert isinstance(df, pd.DataFrame)


@pytest.mark.integration
@pytest.mark.parametrize("model", ["robinson_crusoe_basic"])
def test_return_scalar_for_likelihood(model):
    params, options = process_model_or_seed(model)
    options["n_periods"] = 3

    simulate = get_simulate_func(params, options)
    df = simulate(params)

    log_like = get_log_like_func(params, options, df, return_scalar=True)
    value = log_like(params)

    assert isinstance(value, float)

    log_like_contribs = get_log_like_func(params, options, df, return_scalar=False)
    array = log_like_contribs(params)

    assert isinstance(array, np.ndarray)


@pytest.mark.unit
@pytest.mark.precise
@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(2, 10),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_logsumexp(array):
    expected = special.logsumexp(array)
    result = _logsumexp(array)

    np.testing.assert_allclose(result, expected)
