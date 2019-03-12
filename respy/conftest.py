import pandas
import pytest

from respy.python.solve.solve_auxiliary import StateSpace


@pytest.fixture(autouse=True)
def make_imports_available_in_doctest_namespaces(doctest_namespace):
    """Make imports available in doctest namespaces.

    As a suggestion, we should only include imports which are very common.

    """
    doctest_namespace["pd"] = pandas
    doctest_namespace["StateSpace"] = StateSpace
