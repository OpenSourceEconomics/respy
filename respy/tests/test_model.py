"""Test model generation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.pre_processing.model_processing import process_model_spec
from respy.pre_processing.model_processing import write_out_model_spec
from respy.python.interface import minimal_estimation_interface
from respy.tests.codes.auxiliary import minimal_simulate_observed
from respy.tests.codes.random_model import generate_random_model


@pytest.mark.parametrize("seed", range(10))
def test_equality_of_attributes_and_specs_and_files(seed):
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
