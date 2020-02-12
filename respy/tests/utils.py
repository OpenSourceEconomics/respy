import numpy as np

import respy as rp
from respy.tests.random_model import generate_random_model


def process_model_or_seed(model_or_seed=None, **kwargs):
    if isinstance(model_or_seed, str):
        params, options = rp.get_example_model(model_or_seed, with_data=False)
    elif isinstance(model_or_seed, int):
        np.random.seed(model_or_seed)
        params, options = generate_random_model(**kwargs)
    else:
        raise ValueError

    if "kw_94" in str(model_or_seed):
        options["n_periods"] = 10
    if "kw_97" in str(model_or_seed):
        options["n_periods"] = 5
    elif "kw_2000" in str(model_or_seed):
        options["n_periods"] = 3

    return params, options


def compare_state_space_attributes(attr_1, attr_2, func):
    if isinstance(attr_1, dict):
        out = {}
        for key in attr_1:
            out[key] = func(attr_1[key], attr_2[key])
    else:
        out = func(attr_1, attr_2)

    return out
