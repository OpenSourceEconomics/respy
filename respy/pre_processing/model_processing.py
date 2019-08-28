"""Process model specification files or objects."""
import copy
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from estimagic.optimization.utilities import sdcorr_params_to_matrix

from respy.config import DEFAULT_OPTIONS
from respy.pre_processing.model_checking import validate_options
from respy.pre_processing.model_checking import validate_params

warnings.simplefilter("error", category=pd.errors.PerformanceWarning)


def process_params_and_options(params, options):
    params, optim_paras = _process_params(params)

    extended_options = _process_options(options, params)

    validate_params(params, extended_options)
    validate_options(extended_options)

    return params, optim_paras, extended_options


def _process_params(params):
    params = _read_params(params)
    optim_paras = _parse_parameters(params)

    return params, optim_paras


def _process_options(options, params):
    options = _read_options(options)

    options["n_types"] = len(infer_types(params)) + 1
    if options["n_types"] > 1:
        options["type_covariates"] = sorted(
            params.loc["type_2"].index.get_level_values(0)
        )

    extended_options = {**DEFAULT_OPTIONS, **options}
    extended_options["n_lagged_choices"] = _infer_number_of_lagged_choices(
        extended_options, params
    )
    extended_options = _order_choices(extended_options, params)
    extended_options = _set_defaults_for_choices_with_experience(extended_options)
    extended_options = _set_defaults_for_inadmissible_states(extended_options)

    return extended_options


def _read_params(input_):
    input_ = pd.read_csv(input_) if isinstance(input_, Path) else input_.copy()

    if isinstance(input_, pd.DataFrame):
        if not input_.index.names == ["category", "name"]:
            input_.set_index(["category", "name"], inplace=True)
        params = input_["value"]
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
        options = copy.deepcopy(input_)

    return options


def _order_choices(options, params):
    """Define unique order of choices.

    This function defines a unique order of choices. Choices can be separated in choices
    with experience and wage, with experience but without wage and without experience
    and wage. This distinction is used to create a unique ordering of choices. Within
    each group, we order alphabetically. Then, the order is applied to ``options``.

    """
    choices = set(_infer_choices(params))
    choices_w_exp = set(_infer_choices_with_experience(params, options))
    choices_w_wage = set(_infer_choices_with_prefix(params, "wage_"))
    choices_w_exp_wo_wage = choices_w_exp - choices_w_wage
    choices_wo_exp_wo_wage = choices - choices_w_exp

    options["choices_w_wage"] = sorted(choices_w_wage)
    options["choices_w_exp"] = sorted(choices_w_wage) + sorted(
        choices_w_exp - choices_w_wage
    )
    options["choices_wo_exp"] = sorted(choices_wo_exp_wo_wage)

    # Dictionaries are insertion ordered since Python 3.6+.
    order = (
        options["choices_w_wage"]
        + sorted(choices_w_exp_wo_wage)
        + sorted(choices_wo_exp_wo_wage)
    )
    options["choices"] = {key: options["choices"].get(key, {}) for key in order}

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
        if shares.sum() != 1:
            warnings.warn(
                f"The shares of initial experiences for choice '{choice}' do not sum to"
                " one. Shares are divided by their sum for normalization.",
                category=UserWarning,
            )
            shares = shares / shares.sum()
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
    params : pandas.DataFrame or pandas.Series
        Contains model parameters in column ``"value"``.

    """
    optim_paras = {}

    for quantity in params.index.get_level_values("category").unique():
        quant = params.loc[quantity].to_numpy()
        # Scalars should be scalars, not one-dimensional arrays.
        optim_paras[quantity] = quant[0] if quant.shape == (1,) else quant

    cov = sdcorr_params_to_matrix(optim_paras["shocks"])
    optim_paras["shocks_cholesky"] = np.linalg.cholesky(cov)
    optim_paras.pop("shocks")

    short_meas_error = params.loc["meas_error"]
    n_choices = cov.shape[0]
    meas_error = params.loc["shocks"][:n_choices].copy(deep=True)
    meas_error[:] = 0.0
    meas_error.update(short_meas_error)
    optim_paras["meas_error"] = meas_error.to_numpy()

    if "type_shift" in optim_paras:
        types = infer_types(params)
        n_type_covariates = params.loc[types[0]].shape[0]

        optim_paras["type_prob"] = np.vstack(
            (
                np.zeros(n_type_covariates),
                params.loc[types]
                .sort_index()
                .to_numpy()
                .reshape(len(types), n_type_covariates),
            )
        )
        optim_paras["type_shift"] = np.vstack(
            (
                np.zeros(n_choices),
                optim_paras["type_shift"].reshape(len(types), n_choices),
            )
        )
    else:
        optim_paras["type_shift"] = np.zeros((1, n_choices))

    return optim_paras


def infer_types(params):
    return sorted(
        params.index.get_level_values(0)
        .str.extract(r"(\btype_[0-9]+\b)")[0]
        .dropna()
        .unique()
    )


def _infer_choices_with_experience(params, options):
    covariates = options["covariates"]
    parameters = params.index.get_level_values(1)

    used_covariates = [cov for cov in covariates if cov in parameters]

    matches = []
    for param in parameters:
        matches += re.findall(r"exp_([A-Za-z]*)", param)
    for cov in used_covariates:
        matches += re.findall(r"exp_([A-Za-z]*)", covariates[cov])

    return sorted(list(set(matches)))


def _infer_choices_with_prefix(params, prefix):
    return sorted(
        i[len(prefix) :]
        for i in params.index.get_level_values(0).unique()
        if prefix in i
    )


def _infer_choices(params):
    choices_w_wage = _infer_choices_with_prefix(params, "wage_")
    choices_w_nonpec = _infer_choices_with_prefix(params, "nonpec_")

    return list(set(choices_w_wage) | set(choices_w_nonpec))


def _infer_number_of_lagged_choices(options, params):
    """Infer the maximum lag of choices.

    Notes
    -----
    Once, the probability parameter for lagged choices are moved to the parameters
    (https://github.com/OpenSourceEconomics/respy/issues/212) this function should also
    infer from ``params``.

    Example
    -------
    >>> index = pd.MultiIndex.from_tuples([("name", "covariate")])
    >>> params = pd.DataFrame(index=index)
    >>> options = {
    ...     "covariates": {"covariate": "lagged_choice_2 + lagged_choice_1"},
    ...     "core_state_space_filters": [],
    ... }
    >>> _infer_number_of_lagged_choices(options, params)
    2

    """
    covariates = options["covariates"]
    parameters = params.index.get_level_values(1)

    used_covariates = [cov for cov in covariates if cov in parameters]

    matches = []

    # Look in covariates for lagged choices.
    for cov in used_covariates:
        matches += re.findall(r"lagged_choice_([0-9]+)", str(covariates[cov]))

    # Look in state space filters for lagged choices.
    for filter_ in options["core_state_space_filters"]:
        matches += re.findall(r"lagged_choice_([0-9]+)", filter_)

    n_lagged_choices = 0 if not matches else pd.to_numeric(matches).max()

    return n_lagged_choices
