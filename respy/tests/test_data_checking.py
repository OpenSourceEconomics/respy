import numpy as np
import pytest

import respy as rp
from respy.config import EXAMPLE_MODELS
from respy.pre_processing.data_checking import check_dataset_sim
from respy.pre_processing.data_checking import check_estimation_dataset
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_simulated_data(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = rp.get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    attr = rp.process_model_spec(params_spec, options_spec)

    _, df = rp.simulate(attr)

    check_dataset_sim(attr, df)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_estimation_data(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = rp.get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    attr = rp.process_model_spec(params_spec, options_spec)

    state_space, df = rp.simulate(attr)

    check_estimation_dataset(attr, df)
