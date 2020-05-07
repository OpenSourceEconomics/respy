"""Just intent to wrap mainly existing resources."""
import contextlib
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from respy import RespyCls
from respy import estimate as old_estimate
from respy import simulate as old_simulate
from respy.python.shared.shared_constants import (
    DATA_COLUMNS,
    INDEX_TUPLES,
    MODEL_TO_INI,
)
from respy.python.simulate.simulate_auxiliary import write_out as write_out_data


def simulate(params, model, solution_seed=None, simulation_seed=None):
    """Simulate a model.

    Args:
        params (pd.DataFrame): The index has to correspond to the index tuples in
            respy.python.shared.shared_constants.
        model (str): One of ["example", "kw_data_one", "kw_data_two", "kw_data_three"].
            The corresponding .ini files can be found in the ``example`` directory.
        solution_seed (int): Seed for the monte carlo integral in the emax calculation.
        simulation_seed (int): Seed for the simulation.

    Returns:
        pd.DataFrame: The simulated dataset.

    """
    respy_obj = RespyCls(MODEL_TO_INI[model])
    if solution_seed is not None:
        respy_obj.attr["solution_seed"] = solution_seed
    if simulation_seed is not None:
        respy_obj.attr["simulation_seed"] = simulation_seed

    respy_obj.attr["delta"] = params.loc[("delta", "delta"), "value"]
    x = params["value"].values[1:]
    respy_obj.update_model_paras(x)

    with _temporary_working_directory():
        old_simulate(respy_obj)
        df = pd.read_csv(
            respy_obj.attr["file_sim"],
            names=DATA_COLUMNS,
            delim_whitespace=True,
            na_values=".",
            header=None,
        )

    return df


def get_params(model):
    """Get a params DataFrame for the example models.

    Args:
        model (str): One of ["example", "kw_data_one", "kw_data_two", "kw_data_three"].
            The corresponding .ini files can be found in the ``example`` directory.

    Returns:
        pd.DataFrame: Estimagic and modern respy compatible parameters for the model.

    """
    respy_obj = RespyCls(MODEL_TO_INI[model])

    params = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(INDEX_TUPLES), columns=["value"],
    )

    params_dict = respy_obj.attr["model_paras"]

    params.loc["delta", "value"] = respy_obj.attr["delta"]
    params.loc["wage_a", "value"] = params_dict["coeffs_a"]
    params.loc["wage_b", "value"] = params_dict["coeffs_b"]
    params.loc["nonpec_edu", "value"] = params_dict["coeffs_edu"]
    params.loc["nonpec_home", "value"] = params_dict["coeffs_home"]
    params.loc["shocks_chol", "value"] = params_dict["shocks_cholesky"][
        np.tril_indices(4)
    ]

    params["value"] = params["value"].astype(float)
    return params


def collect_likelihood_data(
    params, data, model, solution_seed=None, estimation_seed=None, tau=None
):
    """Evaluate the negative log likelihood function of a model at params.

    Args:
        params (pd.DataFrame): Estimagic and modern respy compatible parameters for the
            model.
        data (pd.DataFrame): Needs to have the columns specified in
            respy.python.shared.shared_constants.DATA_COLUMNS
        model (str): One of ["example", "kw_data_one", "kw_data_two", "kw_data_three"].
            The corresponding .ini files can be found in the ``example`` directory.
        solution_seed (int): Seed for the monte carlo integral in the emax calculation.
        estimation_seed (int): Seed for the calculation of choice probabilities.
        tau (float): Smoothing parameter for the calculation of choice probabilities.

    Returns:
        dict: Intermediate results of the likelihood evaluation. Contains the following:
            - "old_respy_critvval": The clipped mean negative log likelihood
                contribution used as criterion value in the old respy. This is not
                modified in any way.
            - "likelihood_contributions": The likelihood contributions of respy
                converted to a long form DataFrame that also contains a column for
                "id" and "period".

    """
    data = data[DATA_COLUMNS]

    respy_obj = RespyCls(MODEL_TO_INI[model])

    respy_obj.attr["maxfun"] = 0

    if solution_seed is not None:
        respy_obj.attr["seed_emax"] = solution_seed

    if estimation_seed is not None:
        respy_obj.attr["seed_prob"] = estimation_seed

    if tau is not None:
        respy_obj.attr["tau"] = tau

    respy_obj.attr["delta"] = params.loc[("delta", "delta"), "value"]
    x = params["value"].values[1:]
    respy_obj.update_model_paras(x)

    like_data = {}
    with _temporary_working_directory():
        write_out_data(respy_obj, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            _, crit_val = old_estimate(respy_obj)
        like_data["old_respy_crit_val"] = crit_val

        like_data["likelihood_contributions"] = pd.read_pickle(
            "likelihood_contributions.pickle")

        like_data["systematic_payoffs"] = pd.read_pickle("systematic_payoffs.pickle")

        like_data["continuation_values"] = pd.read_pickle("future_payoffs.pickle")

    return like_data


@contextlib.contextmanager
def _temporary_working_directory():
    """Changes working directory and returns to previous on exit."""
    chars = "abcdfghijklmnopqrstuvwxyz0123456789"
    folder_name = "temp_" + "".join(random.choice(chars) for _ in range(30)) + f"_{os.getpid()}"
    path = Path(".").resolve() / folder_name
    path.mkdir()
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(path)


