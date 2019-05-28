"""Test model generation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from respy.config import EXAMPLE_MODELS
from respy.likelihood import get_criterion_function
from respy.likelihood import get_parameter_vector
from respy.pre_processing.model_checking import check_model_attributes
from respy.pre_processing.model_checking import check_model_parameters
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.pre_processing.model_processing import parameters_to_dictionary
from respy.pre_processing.model_processing import parameters_to_vector
from respy.pre_processing.model_processing import process_model_spec
from respy.pre_processing.model_processing import write_out_model_spec
from respy.shared import get_example_model
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import minimal_simulate_observed


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_equality_of_attributes_and_specs_and_files(model_or_seed):
    if isinstance(model_or_seed, str):
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

    x = get_parameter_vector(attr)
    crit_func = get_criterion_function(attr, df)

    crit_val = crit_func(x)

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
        optim_paras = parameters_to_dictionary(x, is_debug=True)
        args = (optim_paras, num_paras, "all", True)
        x = parameters_to_vector(*args)

    np.testing.assert_allclose(base, x)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_check_model_attributes(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    attr = process_model_spec(params_spec, options_spec)

    check_model_attributes(attr)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_check_model_parameters(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    attr = process_model_spec(params_spec, options_spec)

    check_model_parameters(attr["optim_paras"])


@pytest.mark.wip
@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_yaml_for_options(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    path = np.random.choice([Path("os.yaml"), Path("os.yml")])

    with open(path, "w") as y:
        yaml.dump(options_spec, y)

    with open(path, "r") as y:
        options_spec_ = yaml.safe_load(y)

    assert options_spec == options_spec_
