from respy.tests.random_model import generate_random_model
from respy.shared import get_example_model
import numpy as np


def process_model_or_seed(model_or_seed):
    if isinstance(model_or_seed, str):
        params, options = get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params, options = generate_random_model()

    if "kw_97" in str(model_or_seed):
        options["n_periods"] = 10

    return params, options
