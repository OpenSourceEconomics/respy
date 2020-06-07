import pytest

from respy.config import EXAMPLE_MODELS
from respy.interface import get_example_model
from respy.interface import get_parameter_constraints


@pytest.mark.unit
@pytest.mark.parametrize("model", EXAMPLE_MODELS)
def test_get_example_model(model):
    params, options = get_example_model(model, with_data=False)


@pytest.mark.unit
@pytest.mark.parametrize("model", EXAMPLE_MODELS)
def test_get_example_model_with_sd(model):
    params, options = get_example_model(model, with_data=False, with_sd=True)


@pytest.mark.unit
@pytest.mark.parametrize("model", EXAMPLE_MODELS)
def test_get_parameter_constraints(model):
    get_parameter_constraints(model)
