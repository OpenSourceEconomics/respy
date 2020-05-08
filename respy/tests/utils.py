import numba as nb
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


def apply_to_attributes_of_two_state_spaces(attr_1, attr_2, func):
    """Apply a function to two state space attributes, dense or not.

    Attributes might be `state_space.wages` which can be a dictionary or a Numpy array.

    """
    if isinstance(attr_1, dict) or isinstance(attr_1, nb.typed.typeddict.Dict):
        out = {key: func(attr_1[key], attr_2[key]) for key in attr_1}
    else:
        out = func(attr_1, attr_2)

    return out
