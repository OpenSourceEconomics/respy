"""Test model generation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from respy import EXAMPLE_MODELS
from respy import get_example_model
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.pre_processing.model_processing import process_model_spec
from respy.pre_processing.model_processing import write_out_model_spec
from respy.python.interface import minimal_estimation_interface
from respy.python.shared.shared_auxiliary import distribute_parameters
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import minimal_simulate_observed


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_equality_of_attributes_and_specs_and_files(model_or_seed):
    if isinstance(model_or_seed, Path):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    attr = process_model_spec(params_spec, options_spec)

    params_spec_ = _params_spec_from_attributes(attr)
    options_spec_ = _options_spec_from_attributes(attr)

    assert params_spec.equals(params_spec_)
    assert options_spec == options_spec_

    write_out_model_spec(attr, "example")

    attr_ = process_model_spec(Path("example.csv"), Path("example.json"))

    # Nested dictionaries are hard to compare. Thus, convert to specs again.
    params_spec_ = _params_spec_from_attributes(attr_)
    options_spec_ = _options_spec_from_attributes(attr_)

    pd.testing.assert_frame_equal(params_spec, params_spec_, check_less_precise=15)
    assert options_spec == options_spec_


@pytest.mark.parametrize("seed", range(5))
def test_generate_random_model(seed):
    """Test if random model specifications can be simulated and processed."""
    np.random.seed(seed)

    params_spec, options_spec = generate_random_model()

    attr = process_model_spec(params_spec, options_spec)

    df = minimal_simulate_observed(attr)

    x, crit_val = minimal_estimation_interface(attr, df)

    assert x.shape[0] == params_spec.shape[0]
    assert isinstance(crit_val, float)


@pytest.mark.parametrize("seed", range(10))
def test_back_and_forth_transformation_of_parameter_vector(seed):
    """Testing whether back-and-forth transformation have no effect."""
    np.random.seed(seed)

    num_types = np.random.randint(1, 5)
    num_paras = 53 + (num_types - 1) * 6

    # Create random parameter vector
    base = np.random.uniform(size=num_paras)

    x = base.copy()

    # Apply numerous transformations
    for _ in range(10):
        optim_paras = distribute_parameters(x, is_debug=True)
        args = (optim_paras, num_paras, "all", True)
        x = get_optim_paras(*args)

    np.testing.assert_allclose(base, x)
