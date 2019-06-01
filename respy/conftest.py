import numpy as np
import pandas as pd
import pytest

import respy as rp
from development.testing.regression.run_regression import load_regression_tests
from respy.pre_processing.model_processing import process_model_spec
from respy.solve import create_state_space
from respy.solve import StateSpace
from respy.tests.random_model import generate_random_model


@pytest.fixture(autouse=True)
def make_imports_available_in_doctest_namespaces(doctest_namespace):
    """Make imports available in doctest namespaces.

    As a suggestion, we should only include imports which are very common.

    """
    doctest_namespace["rp"] = rp
    doctest_namespace["pd"] = pd
    doctest_namespace["pyth_create_state_space"] = create_state_space
    doctest_namespace["StateSpace"] = StateSpace
    doctest_namespace["generate_random_model"] = generate_random_model
    doctest_namespace["process_model_spec"] = process_model_spec


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    """Each test is executed with the same random seed."""
    np.random.seed(1423)


@pytest.fixture(scope="function", autouse=True)
def fresh_directory(tmpdir):
    """Each test is executed in a fresh directory."""
    tmpdir.chdir()


@pytest.fixture(scope="session")
def regression_vault():
    """Make regression vault available to tests."""
    return load_regression_tests()
