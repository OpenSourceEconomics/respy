import warnings

import pandas as pd
import yaml

from respy.config import EXAMPLE_MODELS
from respy.config import TEST_RESOURCES_DIR
from respy.data import create_kw_97
from respy.simulate import get_simulate_func

KW_94_CONSTRAINTS = [
    {"loc": "shocks_sdcorr", "type": "sdcorr"},
    {"loc": "lagged_choice_1_edu", "type": "fixed"},
    {"loc": "initial_exp_edu", "type": "fixed"},
    {"loc": "maximum_exp", "type": "fixed"},
]

KW_97_BASIC_CONSTRAINTS = [
    {"loc": "shocks_sdcorr", "type": "sdcorr"},
    {"loc": "initial_exp_school", "type": "fixed"},
    {"loc": "maximum_exp", "type": "fixed"},
]

KW_97_EXTENDED_CONSTRAINTS = KW_97_BASIC_CONSTRAINTS + [
    {"query": "name == 'military_dropout'", "type": "equality"},
    {"query": "name == 'common_hs_graduate'", "type": "equality"},
    {"query": "name == 'common_co_graduate'", "type": "equality"},
    {"loc": "lagged_choice_1_school", "type": "fixed"},
    {"loc": "lagged_choice_1_home", "type": "fixed"},
]


def get_example_model(model, with_data=True):
    """Return parameters, options and data (optional) of an example model.

    Parameters
    ----------
    model : str
        Use arbitrary string to see all available models in traceback.
    with_data : bool
        Whether the accompanying data set should be returned.

    """
    assert model in EXAMPLE_MODELS, f"{model} is not in {EXAMPLE_MODELS}."

    options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
    params = pd.read_csv(
        TEST_RESOURCES_DIR / f"{model}.csv", index_col=["category", "name"]
    )

    if "kw_97" in model and with_data:
        df = (create_kw_97(params, options),)
    elif "kw_94" in model and with_data:
        simulate = get_simulate_func(params, options)
        df = (simulate(params),)
    else:
        df = ()
        if with_data:
            warnings.warn(
                f"No data available for model '{model}'.", category=UserWarning
            )

    return (params, options) + df


def get_parameter_constraints(model):
    if "kw_94" in model:
        constraints = KW_94_CONSTRAINTS
    elif "kw_97_basic" == model:
        constraints = KW_97_BASIC_CONSTRAINTS
    elif "kw_97_extended" == model:
        constraints = KW_97_EXTENDED_CONSTRAINTS

    return constraints
