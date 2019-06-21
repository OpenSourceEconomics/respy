import numpy as np
import pytest

from development.testing.regression.run_regression import load_regression_tests


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    """Each test is executed with the same random seed."""
    np.random.seed(1423)


@pytest.fixture(scope="session")
def regression_vault():
    """Make regression vault available to tests."""
    return load_regression_tests()
