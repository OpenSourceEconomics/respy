"""General interface functions for respy."""
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
    {"loc": "initial_exp_edu_10", "type": "fixed"},
    {"loc": "maximum_exp", "type": "fixed"},
]

KW_97_BASIC_CONSTRAINTS = [
    {"loc": "shocks_sdcorr", "type": "sdcorr"},
    {"loc": "'initial_exp_school' in category", "type": "fixed"},
    {"loc": "maximum_exp", "type": "fixed"},
]

KW_97_EXTENDED_CONSTRAINTS = KW_97_BASIC_CONSTRAINTS + [
    {"query": "name == 'military_dropout'", "type": "equality"},
    {"query": "name == 'common_hs_graduate'", "type": "equality"},
    {"query": "name == 'common_co_graduate'", "type": "equality"},
    {"loc": "lagged_choice_1_school", "type": "fixed"},
    {"loc": "lagged_choice_1_home", "type": "fixed"},
]

KW_2000_CONSTRAINTS = [
    {"loc": "shocks_sdcorr", "type": "sdcorr"},
    {"query": "'type' in category", "type": "fixed"},
    {"loc": "lagged_choice_1_school", "type": "fixed"},
    {"loc": "lagged_choice_1_home", "type": "fixed"},
    {"query": "'initial_exp_school' in category", "type": "fixed"},
    {"loc": "maximum_exp", "type": "fixed"},
    {"loc": "observables", "type": "fixed"},
]

ROBINSON_CRUSOE_CONSTRAINTS = [
    {"loc": "shocks_sdcorr", "type": "sdcorr"},
    {"loc": "lagged_choice_1_hammock", "type": "fixed"},
]


def get_example_model(model, with_data=True):
    """Return parameters, options and data (optional) of an example model.

    Parameters
    ----------
    model : str
        Choose one model name in ``{"robinson_crusoe_basic", "robinson_crusoe_extended",
        kw_94_one", "kw_94_two", "kw_94_three", "kw_97_basic", "kw_97_extended"
        "kw_2000"}``.
    with_data : bool
        Whether the accompanying data set should be returned. For some data sets, real
        data can be provided, for others, a simulated data set will be produced.

    """
    assert model in EXAMPLE_MODELS, f"{model} is not in {EXAMPLE_MODELS}."

    options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
    params = pd.read_csv(
        TEST_RESOURCES_DIR / f"{model}.csv", index_col=["category", "name"]
    )

    if "kw_97" in model and with_data:
        df = (create_kw_97(params, options),)
    elif ("kw_94" in model or "robinson" in model) and with_data:
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
    """Get parameter constraints for the estimation compatible with estimagic.

    For more information, see the `documentation of estimagic
    <https://estimagic.readthedocs.io/en/latest/optimization/constraints/
    constraints.html>`_.

    Parameters
    ----------
    model : str
        Choose one model name in ``{"robinson_crusoe_basic", "robinson_crusoe_extended",
        kw_94_one", "kw_94_two", "kw_94_three", "kw_97_basic", "kw_97_extended"
        "kw_2000"}``.

    Returns
    -------
    constraints : list[dict[str, str]]
        A list of dictionaries specifying constraints.

    Examples
    --------
    >>> constr = rp.get_parameter_constraints("robinson_crusoe_basic")
    >>> constr
    [{'loc': 'shocks_sdcorr', 'type': 'sdcorr'}, ...

    """
    if "kw_94" in model:
        constraints = KW_94_CONSTRAINTS
    elif "kw_97_basic" == model:
        constraints = KW_97_BASIC_CONSTRAINTS
    elif "kw_97_extended" == model:
        constraints = KW_97_EXTENDED_CONSTRAINTS
    elif "kw_2000" == model:
        constraints = KW_2000_CONSTRAINTS
    elif "robinson_crusoe" in model:
        constraints = ROBINSON_CRUSOE_CONSTRAINTS
    else:
        raise NotImplementedError(f"No constraints defined for model {model}.")

    return constraints
