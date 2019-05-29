"""Test model generation."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from respy.config import EXAMPLE_MODELS
from respy.likelihood import get_crit_func_and_initial_guess
from respy.pre_processing.model_checking import check_model_attributes
from respy.pre_processing.model_checking import check_model_parameters
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.pre_processing.model_processing import parse_parameters
from respy.pre_processing.model_processing import process_model_spec
from respy.pre_processing.model_processing import write_out_model_spec
from respy.shared import get_example_model
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_back_and_forth_transformation_of_specs(model_or_seed):
    # 1. Start with params_spec and options_spec.
    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    # 2. Convert to dictionaries.
    attr, optim_paras = process_model_spec(params_spec, options_spec)

    # 3. Convert back to params_spec and options_spec.
    params_spec_ = _params_spec_from_attributes(optim_paras)
    options_spec_ = _options_spec_from_attributes(attr)

    assert params_spec.para.equals(params_spec_.para)
    assert options_spec == options_spec_

    write_out_model_spec(attr, optim_paras, "example")

    attr_, optim_paras_ = process_model_spec(Path("example.csv"), Path("example.json"))

    # Nested dictionaries are hard to compare. Thus, convert to specs again.
    params_spec_ = _params_spec_from_attributes(optim_paras_)
    options_spec_ = _options_spec_from_attributes(attr_)

    params_spec.para.equals(params_spec_.para)
    assert options_spec == options_spec_


@pytest.mark.parametrize("seed", range(5))
def test_generate_random_model(seed):
    """Test if random model specifications can be simulated and processed."""
    np.random.seed(seed)

    params_spec, options_spec = generate_random_model()

    df = simulate_truncated_data(params_spec, options_spec)

    x, crit_func = get_crit_func_and_initial_guess(params_spec, options_spec, df)

    crit_val = crit_func(x)

    assert x.shape[0] == params_spec.shape[0]
    assert isinstance(crit_val, float)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_back_and_forth_transformation_of_parameter_vector(model_or_seed):
    """Testing whether back-and-forth transformation have no effect."""
    # 1. Start with params_spec and options_spec and calculate likelihood.
    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    df = simulate_truncated_data(params_spec, options_spec)

    x, crit_func = get_crit_func_and_initial_guess(params_spec, options_spec, df)
    likelihood = crit_func(x)

    # 2. Convert parameter vector back to params_spec and calculate likelihood again.
    optim_paras_ = parse_parameters(x)
    params_spec_ = _params_spec_from_attributes(optim_paras_)

    x_, crit_func_ = get_crit_func_and_initial_guess(params_spec_, options_spec, df)
    likelihood_ = crit_func_(x_)

    assert likelihood == likelihood_
    np.testing.assert_array_equal(x, x_)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_check_model_attributes_and_parameters(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    attr, optim_paras = process_model_spec(params_spec, options_spec)

    check_model_attributes(attr)
    check_model_parameters(optim_paras)


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
