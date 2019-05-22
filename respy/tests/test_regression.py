"""Run a few regression tests."""
import pytest

from development.testing.regression.run_regression import check_single


params = range(10)


@pytest.mark.parametrize("index", params)
def test_single_regression(regression_vault, index):
    """Run a single regression test."""
    test = regression_vault[index]
    assert check_single(test)
