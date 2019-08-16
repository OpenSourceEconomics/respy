import warnings

import pandas as pd
import yaml

from respy.config import EXAMPLE_MODELS
from respy.config import TEST_RESOURCES_DIR
from respy.data import create_kw_97
from respy.simulate import get_simulate_func


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
        df = (create_kw_97(),)
    elif "kw_94" in model and with_data:
        simulate = get_simulate_func(params, options)
        df = (simulate(params),)
    else:
        df = ()
        warnings.warn(f"No data available for model '{model}'.", category=UserWarning)

    return (params, options) + df
