"""Test model generation."""
import numpy as np
import pytest
from deepdiff import DeepDiff

from respy import get_example_model
from respy.config import EXAMPLE_MODELS
from respy.likelihood import get_crit_func
from respy.pre_processing.model_checking import validate_options
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

    _, _, options = process_params_and_options(_, options)

    validate_options(options)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_invariance_to_order_of_initial_schooling_levels(model_or_seed):
    bound_constr = {"max_edu_start": 10}

    if isinstance(model_or_seed, str):
        params, options = get_example_model(model_or_seed, with_data=False)
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

    assert not DeepDiff(options, shuffled_options)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_invariance_to_order_of_choices(model_or_seed):
    params, options = process_model_or_seed(model_or_seed)

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


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_sorting_of_type_probability_parameters(model_or_seed):
    # Set configuration for random models.
    n_types = np.random.randint(2, 5)
    n_type_covariates = np.random.randint(2, 4)

    params, options = process_model_or_seed(
        model_or_seed, n_types=n_types, n_type_covariates=n_type_covariates
    )

    params, optim_paras, options = process_params_and_options(params, options)

    # Update variables if not a random model, but an example model is tested.
    if isinstance(model_or_seed, str):
        n_types = options["n_types"]
        n_type_covariates = None if n_types == 1 else len(options["type_covariates"])

    if options["n_types"] > 1:
        # Resort type probability parameters.
        types = [f"type_{i}" for i in range(2, options["n_types"] + 1)]
        params.loc[types] = params.sort_index(ascending=False).loc[types]

        _, optim_paras_, _ = process_params_and_options(params, options)

        assert (optim_paras["type_prob"] == optim_paras_["type_prob"]).all()
