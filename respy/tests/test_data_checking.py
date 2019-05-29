import numpy as np
import pytest

import respy as rp
from respy.config import EXAMPLE_MODELS
from respy.pre_processing.data_checking import check_simulated_data
from respy.pre_processing.model_processing import process_model_spec
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(10)))
def test_simulated_data(model_or_seed):
    """Test simulated data with ``check_simulated_data``.

    Note that, ``check_estimation_data`` is also tested in this function as these tests
    focus on a subset of the data.

    """
    if isinstance(model_or_seed, str):
        params_spec, options_spec = rp.get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    _, df = rp.simulate(params_spec, options_spec)

    attr, optim_paras = process_model_spec(params_spec, options_spec)
    check_simulated_data(attr, optim_paras, df)
