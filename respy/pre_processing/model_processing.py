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


def _get_num_types(params_spec):
    if "type_shares" in params_spec.index:
        len_type_shares = len(params_spec.loc["type_shares"])
        num_types = len_type_shares / 2 + 1
    else:
        num_types = 1

    return num_types


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
        paras_vec = params["para"].to_numpy()
    elif isinstance(params, pd.Series):
        paras_vec = params.to_numpy()
    else:
        raise ValueError("Invalid params in parse_parameters: {}.".format(type(params)))

    pinfo = _paras_parsing_information(len(paras_vec))
    optim_paras = {}

    # basic extraction
    for quantity in pinfo:
        start = pinfo[quantity]["start"]
        stop = pinfo[quantity]["stop"]
        optim_paras[quantity] = paras_vec[start:stop]

    # modify the shock_coeffs
    if paras_type == "econ":
        shocks_cholesky = _coeffs_to_cholesky(optim_paras["shocks_coeffs"])
    else:
        shocks_cholesky = _extract_cholesky(paras_vec)
    optim_paras["shocks_cholesky"] = shocks_cholesky
    del optim_paras["shocks_coeffs"]

    # overwrite the type information
    type_shares, type_shifts = _extract_type_information(paras_vec)
    optim_paras["type_shares"] = type_shares
    optim_paras["type_shifts"] = type_shifts
    optim_paras["num_paras"] = paras_vec.shape[0]
    optim_paras["num_types"] = type_shifts.shape[0]

    return optim_paras


def _extract_type_information(x):
    """Extract the information about types from a parameter vector of type 'optim'."""
    pinfo = _paras_parsing_information(len(x))

    # Type shares
    start, stop = pinfo["type_shares"]["start"], pinfo["type_shares"]["stop"]
    num_types = int(len(x[start:]) / 6) + 1
    type_shares = x[start:stop]
    type_shares = np.hstack((np.zeros(2), type_shares))

    # Type shifts
    start, stop = pinfo["type_shifts"]["start"], pinfo["type_shifts"]["stop"]
    type_shifts = x[start:stop]
    type_shifts = np.reshape(type_shifts, (num_types - 1, 4))
    type_shifts = np.vstack((np.zeros(4), type_shifts))

    return type_shares, type_shifts


def _extract_cholesky(x):
    """Extract the Cholesky factor from the shock's covariance matrix."""
    pinfo = _paras_parsing_information(len(x))
    start, stop = (pinfo["shocks_coeffs"]["start"], pinfo["shocks_coeffs"]["stop"])
    shocks_coeffs = x[start:stop]

    dim = _get_matrix_dimension_from_num_triangular_elements(len(shocks_coeffs))
    shocks_cholesky = np.zeros((dim, dim))
    shocks_cholesky[np.tril_indices(dim)] = shocks_coeffs

    return shocks_cholesky


def _coeffs_to_cholesky(coeffs):
    """Return the cholesky factor of a covariance matrix described by coeffs.

    The function can handle the case of a deterministic model where all coefficients
    were zero.

    Parameters
    ----------
    coeffs : np.ndarray
        Array with shape (num_coeffs,) that contains the upper triangular elements of a
        covariance matrix whose diagonal elements have been replaced by their square
        roots.

    """
    dim = _get_matrix_dimension_from_num_triangular_elements(coeffs.shape[0])
    shocks = np.zeros((dim, dim))
    shocks[np.triu_indices(dim)] = coeffs
    shocks[np.diag_indices(dim)] **= 2

    shocks_cov = shocks + shocks.T - np.diag(shocks.diagonal())

    if np.count_nonzero(shocks_cov) == 0:
        return np.zeros((dim, dim))
    else:
        return np.linalg.cholesky(shocks_cov)


def _paras_parsing_information(num_paras):
    """Dictionary with the start and stop indices of each quantity."""
    num_types = int((num_paras - 53) / 6) + 1
    num_shares = (num_types - 1) * 2
    pinfo = {
        "delta": {"start": 0, "stop": 1},
        "coeffs_common": {"start": 1, "stop": 3},
        "coeffs_a": {"start": 3, "stop": 18},
        "coeffs_b": {"start": 18, "stop": 33},
        "coeffs_edu": {"start": 33, "stop": 40},
        "coeffs_home": {"start": 40, "stop": 43},
        "shocks_coeffs": {"start": 43, "stop": 53},
        "type_shares": {"start": 53, "stop": 53 + num_shares},
        "type_shifts": {"start": 53 + num_shares, "stop": num_paras},
    }
    return pinfo


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
