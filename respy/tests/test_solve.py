from pathlib import Path

import numpy as np
import pytest

from respy import EXAMPLE_MODELS
from respy import get_example_model
from respy.interface import minimal_simulation_interface
from respy.pre_processing.model_checking import check_model_solution
from respy.pre_processing.model_processing import process_model_spec
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_check_model_solution(model_or_seed):
    if isinstance(model_or_seed, Path):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    attr = process_model_spec(params_spec, options_spec)

    state_space, _ = minimal_simulation_interface(attr)

    check_model_solution(attr, state_space)
