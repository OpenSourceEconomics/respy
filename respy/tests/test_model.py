"""Test model generation."""
import numpy as np
import pandas as pd
import pytest

from respy.config import EXAMPLE_MODELS
from respy.likelihood import get_crit_func
from respy.pre_processing.model_checking import validate_options
from respy.pre_processing.model_processing import _parse_initial_and_max_experience
from respy.pre_processing.model_processing import process_params_and_options
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data
from respy.tests.utils import process_model_or_seed


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
    _, options = process_model_or_seed(model_or_seed)

    _, options = process_params_and_options(_, options)

    validate_options(options)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_sorting_of_type_probability_parameters(model_or_seed):
    # Set configuration for random models.
    n_types = np.random.randint(2, 5)
    n_type_covariates = np.random.randint(2, 4)

    params, options = process_model_or_seed(
        model_or_seed, n_types=n_types, n_type_covariates=n_type_covariates
    )

    optim_paras, options = process_params_and_options(params, options)

    # Update variables if not a random model, but an example model is tested.
    if isinstance(model_or_seed, str):
        n_types = optim_paras["n_types"]
        n_type_covariates = (
            None if n_types == 1 else len(optim_paras["type_covariates"])
        )

    if optim_paras["n_types"] > 1:
        # Resort type probability parameters.
        types = [f"type_{i}" for i in range(2, optim_paras["n_types"] + 1)]
        params.loc[types] = params.sort_index(ascending=False).loc[types]

        optim_paras_, _ = process_params_and_options(params, options)

        assert (optim_paras["type_prob"] == optim_paras_["type_prob"]).all()


@pytest.mark.wip
def test_parse_initial_and_max_experience():
    """Test ensures that probabilities are transformed with logs and rest passes."""
    choices = ["a", "b"]

    options = {"n_periods": 10}
    optim_paras = {"choices_w_exp": choices, "choices": {"a": {}, "b": {}}}
    params = pd.DataFrame(
        {
            "category": [
                "initial_exp_a_0",
                "initial_exp_a_5",
                "initial_exp_b_0",
                "initial_exp_b_5",
                "maximum_exp",
            ],
            "name": ["probability"] * 2 + ["constant"] * 2 + ["b"],
            "value": [2, 2, np.log(2), np.log(2), 5],
        }
    ).set_index(["category", "name"])["value"]

    optim_paras = _parse_initial_and_max_experience(optim_paras, params, options)

    assert (
        optim_paras["choices"]["a"]["start"][0]
        == optim_paras["choices"]["a"]["start"][5]
    ).all()
    assert (
        optim_paras["choices"]["b"]["start"][0]
        == optim_paras["choices"]["b"]["start"][5]
    ).all()
    assert optim_paras["choices"]["a"]["max"] == options["n_periods"] - 1
    assert optim_paras["choices"]["b"]["max"] == 5
