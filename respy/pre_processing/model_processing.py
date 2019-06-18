"""Process model specification files or objects."""
import collections
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
    params, optim_paras = process_params(params)
    options = process_options(options, params, optim_paras)

    return params, optim_paras, options


def process_params(params):
    params = _read_params(params)
    optim_paras = _parse_parameters(params)

    return params, optim_paras


def process_options(options, params, optim_paras):
    options = _read_options(options)

    for key in DEFAULT_OPTIONS:
        options[key] = options.get(key, DEFAULT_OPTIONS[key])

    options["n_types"] = optim_paras["type_shifts"].shape[0]
    options["choices_w_exp"] = _infer_choices_with_experience(params, options)
    options["choices_w_wage"] = _infer_choices_with_wage(params)

    options = _process_choices(options)

    options = _process_choices_with_experience(options)

    options = _process_inadmissible_states(options)

    _validate_options(options)

    return options


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


def _process_choices(options):
    # Choices can be separated in choices with experience and wage, with experience but
    # without wage and without experience and wage. This distinction is used to create a
    # unique ordering of choices. Within each group, we order alphabetically.
    choices_w_exp_wo_wage = sorted(
        list(set(options["choices_w_exp"]) - set(options["choices_w_wage"]))
    )
    choices_wo_exp_wo_wage = sorted(
        list(set(options["choices"]) - set(options["choices_w_exp"]))
    )

    options["choices_wo_exp"] = choices_wo_exp_wo_wage
    options["choices_wo_wage"] = choices_w_exp_wo_wage + choices_wo_exp_wo_wage

    # We apply the aforementioned order to ``options["choices"]``. As dictionaries are
    # insertion ordered since Python 3.6+, it is recreated.
    order = options["choices_w_wage"] + choices_w_exp_wo_wage + choices_wo_exp_wo_wage
    options["choices"] = {key: options["choices"][key] for key in order}

    return options


def _process_choices_with_experience(o):
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
    choices = o["choices"]
    for choice in o["choices_w_exp"]:
        start = np.array(choices[choice].get("start", [0]))
        ordered_indices = np.argsort(start)
        n_edu_starts = start.shape[0]
        choices[choice]["start"] = start[ordered_indices]
        for key in ["lagged", "share"]:
            default = (
                np.ones(n_edu_starts) / n_edu_starts
                if key == "share"
                else np.zeros(n_edu_starts)
            )
            val = np.array(choices[choice].get(key, default))[ordered_indices]
            # Smooth probabilities so that the sum equals one.
            choices[choice][key] = val / val.sum() if key == "share" else val

        choices[choice]["max"] = choices[choice].get("max", o["n_periods"] - 1)

    o["maximum_exp"] = np.array(
        [choices[choice]["max"] for choice in o["choices_w_exp"]]
    )

    return o


def _process_inadmissible_states(options):
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


def save_options(options, path):
    def _numpy_to_list(d):
        for key, value in d.items():
            if isinstance(value, collections.Mapping):
                d[key] = _numpy_to_list(d[key])
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
            else:
                d[key] = value
        return d

    options = _numpy_to_list(options)

    with open(path, "w") as file:
        yaml.dump(options, file)


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
