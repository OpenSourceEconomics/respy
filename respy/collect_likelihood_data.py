import contextlib
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import respy as rp


def get_data_collecting_crit_func(params, options, old_data):
    data = _prepare_data_for_new_respy(old_data)
    params = _prepare_params_for_new_respy(params)
    normal_crit_func = rp.get_crit_func(params, options, data)

    def data_collecting_crit_func(arg):
        params = arg[0]
        save_path = arg[1]
        params = _prepare_params_for_new_respy(params)
        like_data = {}
        with _temporary_working_directory():
            crit_val = normal_crit_func(params)
            like_data["new_respy_crit_val"] = crit_val

            like_data["likelihood_contributions"] = pd.read_pickle(
                "likelihood_contributions.pickle"
            )

            like_data["systematic_payoffs"] = pd.read_pickle(
                "systematic_payoffs.pickle"
            )

            like_data["continuation_values"] = pd.read_pickle("future_payoffs.pickle")

        if save_path is not None:
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)

            pd.to_pickle(like_data, save_path)

        return like_data

    return data_collecting_crit_func


@contextlib.contextmanager
def _temporary_working_directory():
    """Changes working directory and returns to previous on exit."""
    chars = "abcdfghijklmnopqrstuvwxyz0123456789"
    folder_name = (
        "temp_" + "".join(random.choice(chars) for _ in range(30)) + f"_{os.getpid()}"
    )
    path = Path(".").resolve() / folder_name
    path.mkdir()
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(path)


def _prepare_data_for_new_respy(old_data):
    df = old_data.copy(deep=True)
    df.set_index(["Identifier", "Period"], inplace=True)

    df["Choice"] = pd.Categorical(
        df["Choice"].replace({1: "a", 2: "b", 3: "edu", 4: "home"})
    )

    df["Lagged_Choice_1"] = (
        df.groupby("Identifier")["Choice"].transform("shift").dropna()
    )
    df["Lagged_Choice_1"] = df["Lagged_Choice_1"].where(
        df["Lagged Schooling"] == 0, "edu"
    )

    df.drop(columns="Lagged Schooling", inplace=True)

    return df


def _prepare_params_for_new_respy(old_params):
    params = old_params.copy(deep=True)
    params.index.names = ["category", "name"]
    params.loc[("lagged_choice_1_edu", "probability"), "value"] = 1
    params.loc[("initial_exp_edu_10", "probability"), "value"] = 1
    params.loc[("maximum_exp", "edu"), "value"] = 20
    params.loc[
        ("inadmissibility_penalty", "inadmissibility_penalty"), "value"
    ] = -400_000
    return params
