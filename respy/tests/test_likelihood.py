import numpy as np
import pandas as pd
import pytest

from respy.likelihood import get_crit_func
from respy.simulate import get_simulate_func
from respy.tests.utils import process_model_or_seed


@pytest.mark.integration
@pytest.mark.parametrize("model", ["kw_94_one", "kw_97_basic"])
def test_return_comparison_plot_data_for_likelihood(model):
    params, options = process_model_or_seed(model)

    simulate = get_simulate_func(params, options)
    df = simulate(params)

    loglike = get_crit_func(params, options, df, return_comparison_plot_data=False)
    loglike = loglike(params)

    assert isinstance(loglike, float)

    loglike = get_crit_func(params, options, df, return_comparison_plot_data=True)
    loglike, df = loglike(params)

    assert isinstance(loglike, float)
    assert isinstance(df, pd.DataFrame)


@pytest.mark.integration
@pytest.mark.parametrize("model", ["kw_94_one", "kw_97_basic"])
def test_return_scalar_for_likelihood(model):
    params, options = process_model_or_seed(model)

    simulate = get_simulate_func(params, options)
    df = simulate(params)

    loglike = get_crit_func(params, options, df, return_scalar=True)
    value = loglike(params)

    assert isinstance(value, float)

    loglike = get_crit_func(params, options, df, return_scalar=False)
    array = loglike(params)

    assert isinstance(array, np.ndarray)
