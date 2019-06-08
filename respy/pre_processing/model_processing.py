"""Process model specification files or objects."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from respy.config import DEFAULT_OPTIONS
from respy.pre_processing.model_checking import _validate_options

warnings.simplefilter("error", category=pd.errors.PerformanceWarning)


def process_params(params):
    params = _read_params(params)
    optim_paras = parse_parameters(params)

    return params, optim_paras


def process_options(options):
    options = _read_options(options)

    for key in DEFAULT_OPTIONS:
        if key in ["covariates", "inadmissible_states"]:
            options[key] = {**DEFAULT_OPTIONS[key], **options.get(key, {})}
        else:
            options[key] = options.get(key, DEFAULT_OPTIONS[key])

    _validate_options(options)
    options = _sort_education_options(options)

    return options


def _sort_education_options(o):
    ordered_indices = np.argsort(o["education_start"])
    for key in ["education_start", "education_share", "education_lagged"]:
        o[key] = np.array(o[key])[ordered_indices].tolist()
    return o


def _read_params(input_):
    if not isinstance(input_, (Path, pd.DataFrame)):
        raise TypeError("params_spec must be Path or pd.DataFrame.")

    params_spec = pd.read_csv(input_) if isinstance(input_, Path) else input_

    params_spec["para"] = params_spec["para"].astype(float)

    if not params_spec.index.names == ["category", "name"]:
        params_spec.set_index(["category", "name"], inplace=True)

    return params_spec


def _read_options(input_):
    if not isinstance(input_, (Path, dict)):
        raise TypeError("options_spec must be Path or dictionary.")

    if isinstance(input_, Path):
        with open(input_, "r") as file:
            if input_.suffix in [".yaml", ".yml"]:
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
        raise TypeError(f"Invalid type {type(params)} for params.")

    optim_paras = {}

    for quantity in params.index.get_level_values("category").unique():
        optim_paras[quantity] = params.loc[quantity].to_numpy()

    cov = sdcorr_params_to_matrix(optim_paras["shocks"])
    optim_paras["shocks_cholesky"] = np.linalg.cholesky(cov)
    optim_paras.pop("shocks")

    short_meas_error = params.loc["meas_error"]
    num_choices = cov.shape[0]
    meas_error = params.loc["shocks"][:num_choices].copy(deep=True)
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
        optim_paras["num_types"] = optim_paras["type_shifts"].shape[0]
    else:
        optim_paras["num_types"] = 1
        optim_paras["type_shares"] = np.zeros(2)
        optim_paras["type_shifts"] = np.zeros((1, 4))

    optim_paras["num_paras"] = len(params)

    return optim_paras


def cov_matrix_to_sdcorr_params(cov):
    """Can be taken from estimagic once 0.0.5 is released."""
    dim = len(cov)
    sds = np.sqrt(np.diagonal(cov))
    scaling_matrix = np.diag(1 / sds)
    corr = scaling_matrix.dot(cov).dot(scaling_matrix)
    correlations = corr[np.tril_indices(dim, k=-1)]
    return np.hstack([sds, correlations])


def sdcorr_params_to_matrix(sdcorr_params):
    """Can be taken from estimagic once 0.0.5 is released."""
    dim = number_of_triangular_elements_to_dimension(len(sdcorr_params))
    diag = np.diag(sdcorr_params[:dim])
    lower = np.zeros((dim, dim))
    lower[np.tril_indices(dim, k=-1)] = sdcorr_params[dim:]
    corr = np.eye(dim) + lower + lower.T
    cov = diag.dot(corr).dot(diag)
    return cov


def number_of_triangular_elements_to_dimension(num):
    """Can be taken from estimagic once 0.0.5 is released."""
    return int(np.sqrt(8 * num + 1) / 2 - 0.5)
