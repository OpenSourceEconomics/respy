"""Run a few regression tests."""
import numpy as np
import pytest

from development.testing.regression import calc_crit_val
from respy.config import TOL_REGRESSION_TESTS


@pytest.mark.parametrize("index", range(10))
def test_single_regression(regression_vault, index):
    """Run a single regression test."""
    params, options, exp_val = regression_vault[index]
    crit_val = calc_crit_val(params, options)

    assert np.isclose(
        crit_val, exp_val, rtol=TOL_REGRESSION_TESTS, atol=TOL_REGRESSION_TESTS
    )
