"""Process model specification files or objects."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def process_model_spec(params_spec, options_spec):
    params_spec = _read_params_spec(params_spec)
    options_spec = _read_options_spec(options_spec)

    attr = _create_attribute_dictionary(options_spec)
    optim_paras = parse_parameters(params_spec)

    return attr, optim_paras


def _create_attribute_dictionary(options_spec):
    attr = {
        "is_debug": bool(options_spec["program"]["debug"]),
        "interpolation": bool(options_spec["interpolation"]["flag"]),
        "num_agents_sim": int(options_spec["simulation"]["agents"]),
        "num_draws_sol": int(options_spec["solution"]["draws"]),
        "num_draws_est": int(options_spec["estimation"]["draws"]),
        "num_points_interp": int(options_spec["interpolation"]["points"]),
        "seed_sol": int(options_spec["solution"]["seed"]),
        "seed_est": int(options_spec["estimation"]["seed"]),
        "seed_sim": int(options_spec["simulation"]["seed"]),
        "tau": float(options_spec["estimation"]["tau"]),
        "edu_spec": options_spec["edu_spec"],
        "num_periods": int(options_spec["num_periods"]),
    }

    return attr


def _read_params_spec(input_):
    if not isinstance(input_, (Path, pd.DataFrame)):
        raise TypeError("params_spec must be Path or pd.DataFrame.")

    params_spec = pd.read_csv(input_) if isinstance(input_, Path) else input_

    params_spec["para"] = params_spec["para"].astype(float)

    if not params_spec.index.names == ["category", "name"]:
        params_spec.set_index(["category", "name"], inplace=True)

    return params_spec


def _read_options_spec(input_):
    if not isinstance(input_, (Path, dict)):
        raise TypeError("options_spec must be Path or dictionary.")

    if isinstance(input_, Path):
        with open(input_, "r") as file:
            if input_.suffix == ".json":
                options_spec = json.load(file)
            elif input_.suffix in [".yaml", ".yml"]:
                options_spec = yaml.safe_load(file)
            else:
                raise NotImplementedError(f"Format {input_.suffix} is not supported.")
    else:
        options_spec = input_

    return options_spec


def parse_parameters(params, paras_type="optim"):
    """Parse the parameter vector into a dictionary of model quantities.

    Parameters
    ----------
    params : DataFrame or Series
        DataFrame with parameter specification or 'para' column thereof
    is_debug : bool
        If true, the parameters are checked for validity
    info : ???
        Unknown argument.
    paras_type : str
        one of ['econ', 'optim']. A paras_vec of type 'econ' contains the the standard
        deviations and covariances of the shock distribution. This is how parameters are
        represented in the .ini file and the output of .fit(). A paras_vec of type
        'optim' contains the elements of the cholesky factors of the covariance matrix
        of the shock distribution. This type is used internally during the likelihood
        estimation. The default value is 'optim' in order to make the function more
        aligned with Fortran, where we never have to parse 'econ' parameters.

    """
    if isinstance(params, pd.DataFrame):
        params = params["para"]
    elif isinstance(params, pd.Series):
        pass
    else:
        raise ValueError("Invalid params in parse_parameters: {}.".format(type(params)))

    optim_paras = {}

    for quantity in params.index.get_level_values("category").unique():
        optim_paras[quantity] = params.loc[quantity].to_numpy()

    shocks = optim_paras["shocks"]
    dim = _get_matrix_dimension_from_num_triangular_elements(len(shocks))
    shocks_cholesky = np.zeros((dim, dim))
    shocks_cholesky[np.tril_indices(dim)] = shocks
    optim_paras["shocks_cholesky"] = shocks_cholesky

    if "type_shares" in optim_paras:
        optim_paras["type_shares"] = np.hstack(
            [np.zeros(2), optim_paras["type_shares"]]
        )
        optim_paras["type_shifts"] = np.vstack(
            [np.zeros(4), optim_paras["type_shift"].reshape(-1, 4)]
        )
        optim_paras["num_types"] = optim_paras["type_shifts"].shape[0]
    else:
        optim_paras["num_types"] = 1
        optim_paras["type_shares"] = np.zeros(2)
        optim_paras["type_shifts"] = np.zeros((1, 4))

    optim_paras["num_paras"] = len(params)

    return optim_paras


def _get_matrix_dimension_from_num_triangular_elements(num):
    """Calculate the dimension of a square matrix from number of triangular elements.

    Parameters
    ----------
    num : int
        The number of upper or lower triangular elements in the matrix.

    Example
    -------
    >>> _get_matrix_dimension_from_num_triangular_elements(6)
    3
    >>> _get_matrix_dimension_from_num_triangular_elements(10)
    4

    """
    return int(np.sqrt(8 * num + 1) / 2 - 0.5)
