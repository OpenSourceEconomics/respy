"""Process model specification files or objects."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from respy.config import TINY_FLOAT
from respy.pre_processing.model_checking import _check_parameter_vector
from respy.pre_processing.model_checking import check_model_parameters
from respy.pre_processing.specification_helpers import csv_template
from respy.shared import _paras_parsing_information
from respy.shared import number_of_triangular_elements_to_dimensio


def process_model_spec(params_spec, options_spec):
    params_spec = _read_params_spec(params_spec)
    options_spec = _read_options_spec(options_spec)
    attr = _create_attribute_dictionary(params_spec, options_spec)

    return attr


def write_out_model_spec(attr, save_path):
    params_spec = _params_spec_from_attributes(attr)
    options_spec = _options_spec_from_attributes(attr)

    params_spec.to_csv(Path(save_path).with_suffix(".csv"))
    with open(Path(save_path).with_suffix(".json"), "w") as j:
        json.dump(options_spec, j)


def _options_spec_from_attributes(attr):
    estimation = {
        "draws": attr["num_draws_prob"],
        "agents": attr["num_agents_est"],
        "seed": attr["seed_prob"],
        "tau": attr["tau"],
    }

    simulation = {"agents": attr["num_agents_sim"], "seed": attr["seed_sim"]}

    program = {"debug": attr["is_debug"]}

    interpolation = {"flag": attr["interpolation"], "points": attr["num_points_interp"]}

    solution = {"seed": attr["seed_emax"], "draws": attr["num_draws_emax"]}

    options_spec = {
        "estimation": estimation,
        "simulation": simulation,
        "program": program,
        "interpolation": interpolation,
        "solution": solution,
        "edu_spec": attr["edu_spec"],
        "num_periods": attr["num_periods"],
    }

    return options_spec


def _params_spec_from_attributes(attr):
    csv = csv_template(attr["num_types"])
    bounds = np.array(attr["optim_paras"]["paras_bounds"])
    csv["lower"] = bounds[:, 0]
    csv["upper"] = bounds[:, 1]
    csv[["lower", "upper"]] = csv[["lower", "upper"]].astype(float)
    csv["fixed"] = attr["optim_paras"]["paras_fixed"]
    csv["para"] = parameters_to_vector(
        paras_dict=attr["optim_paras"],
        num_paras=attr["num_paras"],
        which="all",
        is_debug=True,
    )
    return csv


def _create_attribute_dictionary(params_spec, options_spec):
    attr = {
        "is_debug": bool(options_spec["program"]["debug"]),
        "interpolation": bool(options_spec["interpolation"]["flag"]),
        "num_agents_est": int(options_spec["estimation"]["agents"]),
        "num_agents_sim": int(options_spec["simulation"]["agents"]),
        "num_draws_emax": int(options_spec["solution"]["draws"]),
        "num_draws_prob": int(options_spec["estimation"]["draws"]),
        "num_points_interp": int(options_spec["interpolation"]["points"]),
        "num_types": int(_get_num_types(params_spec)),
        "optim_paras": parameters_to_dictionary(
            params_spec["para"].to_numpy(), is_debug=True
        ),
        "seed_emax": int(options_spec["solution"]["seed"]),
        "seed_prob": int(options_spec["estimation"]["seed"]),
        "seed_sim": int(options_spec["simulation"]["seed"]),
        "tau": float(options_spec["estimation"]["tau"]),
        "edu_spec": options_spec["edu_spec"],
        "num_periods": int(options_spec["num_periods"]),
        "num_paras": len(params_spec),
        "myopia": params_spec.loc[("delta", "delta"), "para"] == 0.0,
    }

    bounds = []
    for coeff in params_spec.index:
        bound = []
        for bounds_type in ["lower", "upper"]:
            if pd.isnull(params_spec.loc[coeff, bounds_type]):
                bound.append(None)
            else:
                bound.append(float(params_spec.loc[coeff, bounds_type]))
        bounds.append(bound)

    attr["optim_paras"]["paras_bounds"] = bounds
    attr["optim_paras"]["paras_fixed"] = (
        params_spec["fixed"].astype(bool).to_numpy().tolist()
    )

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
            elif input_.suffix == ".yaml":
                options_spec = yaml.load(file)
            else:
                raise NotImplementedError(f"Format {input_.suffix} is not supported.")
    else:
        options_spec = input_

    return options_spec


def parameters_to_dictionary(paras_vec, is_debug=False, info=None, paras_type="optim"):
    """Parse the parameter vector into a dictionary of model quantities.

    Parameters
    ----------
    paras_vec : np.ndarray
        1d numpy array with the parameters
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
    paras_vec = paras_vec.copy()
    assert paras_type in ["econ", "optim"], "paras_type must be econ or optim."

    if is_debug and paras_type == "optim":
        _check_parameter_vector(paras_vec)

    pinfo = _paras_parsing_information(len(paras_vec))
    paras_dict = {}

    # basic extraction
    for quantity in pinfo:
        start = pinfo[quantity]["start"]
        stop = pinfo[quantity]["stop"]
        paras_dict[quantity] = paras_vec[start:stop]

    # modify the shock_coeffs
    if paras_type == "econ":
        shocks_cholesky = _coeffs_to_cholesky(paras_dict["shocks_coeffs"])
    else:
        shocks_cholesky, info = _extract_cholesky(paras_vec, info)
    paras_dict["shocks_cholesky"] = shocks_cholesky
    del paras_dict["shocks_coeffs"]

    # overwrite the type information
    type_shares, type_shifts = _extract_type_information(paras_vec)
    paras_dict["type_shares"] = type_shares
    paras_dict["type_shifts"] = type_shifts

    # checks
    if is_debug:
        assert check_model_parameters(paras_dict)

    return paras_dict


def parameters_to_vector(paras_dict, num_paras, which, is_debug):
    """Stack optimization parameters from a dictionary into a vector of type 'optim'.

    Parameters
    ----------
    paras_dict : dict
        dictionary with quantities from which the parameters can be extracted.
    num_paras : int
        number of parameters in the model (not only free parameters)
    which : str
        one of ['free', 'all'], determines whether the resulting parameter vector
        contains only free parameters or all parameters.
    is_debug : bool
        If True, inputs and outputs are checked for consistency.

    """
    if is_debug:
        assert which in ["free", "all"], 'which must be in ["free", "all"]'
        assert check_model_parameters(paras_dict)

    pinfo = _paras_parsing_information(num_paras)
    x = np.full(num_paras, np.nan)

    start, stop = pinfo["delta"]["start"], pinfo["delta"]["stop"]
    x[start:stop] = paras_dict["delta"]

    start, stop = (pinfo["coeffs_common"]["start"], pinfo["coeffs_common"]["stop"])
    x[start:stop] = paras_dict["coeffs_common"]

    start, stop = pinfo["coeffs_a"]["start"], pinfo["coeffs_a"]["stop"]
    x[start:stop] = paras_dict["coeffs_a"]

    start, stop = pinfo["coeffs_b"]["start"], pinfo["coeffs_b"]["stop"]
    x[start:stop] = paras_dict["coeffs_b"]

    start, stop = pinfo["coeffs_edu"]["start"], pinfo["coeffs_edu"]["stop"]
    x[start:stop] = paras_dict["coeffs_edu"]

    start, stop = pinfo["coeffs_home"]["start"], pinfo["coeffs_home"]["stop"]
    x[start:stop] = paras_dict["coeffs_home"]

    start, stop = (pinfo["shocks_coeffs"]["start"], pinfo["shocks_coeffs"]["stop"])
    x[start:stop] = paras_dict["shocks_cholesky"][np.tril_indices(4)]

    start, stop = pinfo["type_shares"]["start"], pinfo["type_shares"]["stop"]
    x[start:stop] = paras_dict["type_shares"][2:]

    start, stop = pinfo["type_shifts"]["start"], pinfo["type_shifts"]["stop"]
    x[start:stop] = paras_dict["type_shifts"].flatten()[4:]

    if is_debug:
        _check_parameter_vector(x)

    if which == "free":
        x = [x[i] for i in range(num_paras) if not paras_dict["paras_fixed"][i]]
        x = np.array(x)

    return x


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


def _extract_cholesky(x, info=None):
    """Extract the cholesky factor of the shock covariance from parameters of type
    'optim."""
    pinfo = _paras_parsing_information(len(x))
    start, stop = (pinfo["shocks_coeffs"]["start"], pinfo["shocks_coeffs"]["stop"])
    shocks_coeffs = x[start:stop]
    dim = number_of_triangular_elements_to_dimensio(len(shocks_coeffs))
    shocks_cholesky = np.zeros((dim, dim))
    shocks_cholesky[np.tril_indices(dim)] = shocks_coeffs

    # Stabilization
    if info is not None:
        info = 0

    # We need to ensure that the diagonal elements are larger than zero during
    # estimation. However, we want to allow for the special case of total
    # absence of randomness for testing with simulated datasets.
    if not (np.count_nonzero(shocks_cholesky) == 0):
        shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
        for i in range(len(shocks_cov)):
            if np.abs(shocks_cov[i, i]) < TINY_FLOAT:
                shocks_cholesky[i, i] = np.sqrt(TINY_FLOAT)
                if info is not None:
                    info = 1

    if info is not None:
        return shocks_cholesky, info
    else:
        return shocks_cholesky, None


def _coeffs_to_cholesky(coeffs):
    """Return the cholesky factor of a covariance matrix described by coeffs.

    The function can handle the case of a deterministic model (i.e. where all coeffs =
    0)

    Args:
        coeffs (np.ndarray): 1d numpy array that contains the upper triangular elements
            of a covariance matrix whose diagonal elements have been replaced by their
            square roots.

    """
    dim = number_of_triangular_elements_to_dimensio(coeffs.shape[0])
    shocks = np.zeros((dim, dim))
    shocks[np.triu_indices(dim)] = coeffs
    shocks[np.diag_indices(dim)] **= 2

    shocks_cov = shocks + shocks.T - np.diag(shocks.diagonal())

    if np.count_nonzero(shocks_cov) == 0:
        return np.zeros((dim, dim))
    else:
        return np.linalg.cholesky(shocks_cov)
