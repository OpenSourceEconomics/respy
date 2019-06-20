"""Process model specification files or objects."""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from estimagic.optimization.utilities import sdcorr_params_to_matrix

from respy.config import DEFAULT_OPTIONS
from respy.pre_processing.model_checking import _validate_options

warnings.simplefilter("error", category=pd.errors.PerformanceWarning)


def process_params_and_options(params, options):
    params, optim_paras = _process_params(params)

    n_types = optim_paras["type_shifts"].shape[0]
    extended_options = _process_options(options, params, n_types)

    return params, optim_paras, extended_options


def _process_params(params):
    params = _read_params(params)
    optim_paras = _parse_parameters(params)

    return params, optim_paras


def _process_options(options, params, n_types):
    options = _read_options(options)

    extended_options = {**DEFAULT_OPTIONS, **options}
    extended_options["n_types"] = n_types
    extended_options["choices_w_exp"] = _infer_choices_with_experience(
        params, extended_options
    )
    extended_options["choices_w_wage"] = _infer_choices_with_wage(params)
    extended_options = _order_choices(extended_options)
    extended_options = _set_defaults_for_choices_with_experience(extended_options)
    extended_options = _set_defaults_for_inadmissible_states(extended_options)

    _validate_options(extended_options)

    return extended_options


def _read_params(input_):
    input_ = pd.read_csv(input_) if isinstance(input_, Path) else input_

    if isinstance(input_, pd.DataFrame):
        if not input_.index.names == ["category", "name"]:
            input_.set_index(["category", "name"], inplace=True)
        params = input_["para"]
    elif isinstance(input_, pd.Series):
        params = input_
        assert params.index.names == [
            "category",
            "name",
        ], "params as pd.Series has wrong index."
    else:
        raise TypeError("params must be Path, pd.DataFrame or pd.Series.")

    return params


def _read_options(input_):
    if not isinstance(input_, (Path, dict)):
        raise TypeError("options must be pathlib.Path or dictionary.")

    if isinstance(input_, Path):
        with open(input_, "r") as file:
            if input_.suffix in [".yaml", ".yml"]:
                options = yaml.safe_load(file)
            else:
                raise NotImplementedError(f"Format {input_.suffix} is not supported.")
    else:
        options = input_

    return options


def _order_choices(options):
    """Define unique order of choices.

    This function defines a unique order of choices. Choices can be separated in choices
    with experience and wage, with experience but without wage and without experience
    and wage. This distinction is used to create a unique ordering of choices. Within
    each group, we order alphabetically. Then, the order is applied to ``options``.

    """
    choices_w_exp_wo_wage = sorted(
        list(set(options["choices_w_exp"]) - set(options["choices_w_wage"]))
    )
    choices_wo_exp_wo_wage = sorted(
        list(set(options["choices"]) - set(options["choices_w_exp"]))
    )

    options["choices_wo_exp"] = choices_wo_exp_wo_wage
    options["choices_wo_wage"] = choices_w_exp_wo_wage + choices_wo_exp_wo_wage

    # Dictionaries are insertion ordered since Python 3.6+.
    order = options["choices_w_wage"] + choices_w_exp_wo_wage + choices_wo_exp_wo_wage
    options["choices"] = {key: options["choices"][key] for key in order}

    return options


def _set_defaults_for_choices_with_experience(options):
    """Process initial experience distributions.

    A choice might have information on the distribution of initial experiences which is
    used at the beginning of the simulation to determine the starting points of agents.
    This function makes the model invariant to the order or misspecified probabilities.

    - ``"start"`` determines initial experience levels. Default is to start with zero
      experience.
    - ``"share"`` determines the share of each initial experience level in the starting
      population. Default is a uniform distribution over all initial experience levels.
    - ``"lagged"`` determines the share of the population with this initial experience
      to have this choice as an initial lagged choice. Default is a probability of zero.

    """
    choices = options["choices"]

    for choice in options["choices_w_exp"]:

        starts = np.array(choices[choice].get("start", [0]))
        ordered_indices = np.argsort(starts)
        choices[choice]["start"] = starts[ordered_indices]

        n_starts = starts.shape[0]

        lagged = np.array(choices[choice].get("lagged", np.zeros(n_starts)))
        choices[choice]["lagged"] = lagged[ordered_indices]

        shares = np.array(choices[choice].get("share", np.ones(n_starts)))
        shares /= shares.sum()
        choices[choice]["share"] = shares[ordered_indices]

        choices[choice]["max"] = choices[choice].get("max", options["n_periods"] - 1)

    return options


def _set_defaults_for_inadmissible_states(options):
    for choice in options["choices"]:
        if choice in options["inadmissible_states"]:
            pass
        else:
            options["inadmissible_states"][choice] = "False"

    return options


def _parse_parameters(params):
    """Parse the parameter vector into a dictionary of model quantities.

    Parameters
    ----------
    params : DataFrame or Series
        DataFrame with parameter specification or 'para' column thereof

    """
    optim_paras = {}

    for quantity in params.index.get_level_values("category").unique():
        optim_paras[quantity] = params.loc[quantity].to_numpy()

    cov = sdcorr_params_to_matrix(optim_paras["shocks"])
    optim_paras["shocks_cholesky"] = np.linalg.cholesky(cov)
    optim_paras.pop("shocks")

    short_meas_error = params.loc["meas_error"]
    n_choices = cov.shape[0]
    meas_error = params.loc["shocks"][:n_choices].copy(deep=True)
    meas_error[:] = 0.0
    meas_error.update(short_meas_error)
    optim_paras["meas_error"] = meas_error.to_numpy()

    if "type_shares" in optim_paras:
        optim_paras["type_shares"] = np.hstack(
            [np.zeros(2), optim_paras["type_shares"]]
        )
        optim_paras["type_shifts"] = np.vstack(
            [np.zeros(4), optim_paras["type_shift"].reshape(-1, 4)]
        )
    else:
        optim_paras["type_shares"] = np.zeros(2)
        optim_paras["type_shifts"] = np.zeros((1, 4))

    return optim_paras


def _infer_choices_with_experience(params, options):
    covariates = options["covariates"]
    parameters = params.index.get_level_values(1)

    used_covariates = [cov for cov in covariates if cov in parameters]

    matches = []
    for cov in used_covariates:
        matches += re.findall(r"exp_([A-Za-z]*)", covariates[cov])

    return sorted(list(set(matches)))


def _infer_choices_with_wage(params):
    return sorted(
        i[5:] for i in params.index.get_level_values(0).unique() if "wage_" in i
    )
