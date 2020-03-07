import pytest

from respy import test
from respy.config import EXAMPLE_MODELS
from respy.interface import get_example_model
from respy.interface import get_parameter_constraints


def test_internal_test_command():
    test("--collect-only")


@pytest.mark.parametrize("model", EXAMPLE_MODELS)
def test_get_example_model(model):
    params, options = get_example_model(model, with_data=False)


@pytest.mark.parametrize("model", EXAMPLE_MODELS)
def test_get_parameter_constraints(model):
    assert get_parameter_constraints(model)
