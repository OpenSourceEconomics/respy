"""Run a few regression tests."""
import pytest

from development.testing.regression.run_regression import check_single


@pytest.mark.parametrize("index", range(10))
def test_single_regression(regression_vault, index):
    """Run a single regression test."""
    test = regression_vault[index]
    assert check_single(test)
