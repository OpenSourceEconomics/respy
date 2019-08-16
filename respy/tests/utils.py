import numpy as np

import respy as rp
from respy.tests.random_model import generate_random_model


def process_model_or_seed(model_or_seed, **kwargs):
    if isinstance(model_or_seed, str):
        params, options = rp.get_example_model(model_or_seed, with_data=False)
    else:
        np.random.seed(model_or_seed)
        params, options = generate_random_model(**kwargs)

    if "kw_97" in str(model_or_seed):
        options["n_periods"] = 10

    return params, options
