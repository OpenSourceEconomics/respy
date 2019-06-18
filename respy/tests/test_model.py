"""Test model generation."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from respy.config import EXAMPLE_MODELS
from respy.likelihood import get_crit_func
from respy.pre_processing.model_checking import _validate_options
from respy.pre_processing.model_processing import process_params_and_options
from respy.pre_processing.model_processing import save_options
from respy.shared import get_example_model
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data


@pytest.mark.parametrize("seed", range(5))
def test_generate_random_model(seed):
    """Test if random model specifications can be simulated and processed."""
    np.random.seed(seed)

    params, options = generate_random_model()

    df = simulate_truncated_data(params, options)

    crit_func = get_crit_func(params, options, df)

    crit_val = crit_func(params)

    assert isinstance(crit_val, float)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_model_options(model_or_seed):
    if isinstance(model_or_seed, str):
        _, options = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        _, options = generate_random_model()

    _, _, options = process_params_and_options(_, options)

    _validate_options(options)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_yaml_for_options(model_or_seed):
    if isinstance(model_or_seed, str):
        params, options = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params, options = generate_random_model()

    path = np.random.choice([Path("os.yaml"), Path("os.yml")])

    save_options(options, path)

    with open(path, "r") as y:
        options_ = yaml.safe_load(y)

    assert options == options_


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_invariance_to_order_of_initial_schooling_levels(model_or_seed):
    bound_constr = {"max_edu_start": 10}

    if isinstance(model_or_seed, str):
        params, options = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params, options = generate_random_model(bound_constr=bound_constr)

    shuffled_options = options.copy()
    n_init_levels = len(options["choices"]["edu"]["start"])

    shuffled_order = np.random.choice(n_init_levels, size=n_init_levels, replace=False)

    for label in ["start", "lagged", "share"]:
        shuffled_options["choices"]["edu"][label] = np.array(
            shuffled_options["choices"]["edu"].pop(label)
        )[shuffled_order]

    _, _, options = process_params_and_options(params, options)
    _, _, shuffled_options = process_params_and_options(params, options)

    assert options == shuffled_options


@pytest.mark.wip
@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_invariance_to_order_of_choices(model_or_seed):
    if isinstance(model_or_seed, str):
        params, options = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params, options = generate_random_model()

    shuffled_choices = list(options["choices"].keys())
    np.random.shuffle(shuffled_choices)

    shuffled_options = options.copy()
    shuffled_options["choices"] = {
        choice: shuffled_options["choices"].get(choice, {})
        for choice in shuffled_choices
    }

    _, _, options = process_params_and_options(params, options)
    _, _, shuffled_options = process_params_and_options(params, shuffled_options)

    assert list(options["choices"]) == list(shuffled_options["choices"])
