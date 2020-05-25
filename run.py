import os

n_threads = 1
os.environ["NUMBA_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"

import respy as rp
from respy.collect_likelihood_data import get_data_collecting_crit_func
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import joblib


def make_arg_list(params, options, dataset_name, n_points=31):
    dirname = _make_folder_name_from_options_and_dataset_name(options, dataset_name)
    params = params.copy(deep=True)
    params["lower"] = -0.1
    params["upper"] = 0.1
    params["lower"] = params["lower"].where(
        params["value"] == 0, params["value"] * 0.95
    )
    params["upper"] = params["upper"].where(
        params["value"] == 0, params["value"] * 1.05
    )

    params.loc[("shocks_chol", "chol_edu"), "lower"] = 1350
    params.loc[("shocks_chol", "chol_edu"), "upper"] = 1650

    params.loc[("shocks_chol", "chol_home"), "lower"] = 1350
    params.loc[("shocks_chol", "chol_home"), "upper"] = 1650

    params.loc[("wage_a", "exp_b_square"), "lower"] = -0.03
    params.loc[("wage_a", "exp_b_square"), "upper"] = 0.03

    args_list = []

    for loc in params.index:
        grid = np.linspace(params.loc[loc, "lower"], params.loc[loc, "upper"], n_points)
        for point in grid:
            new_params = params.copy(deep=True)
            new_params.loc[loc, "value"] = point

            name = dirname / f"{loc[0]}-{loc[1]}__{point}.pickle"
            arg = (new_params, name)
            args_list.append(arg)
    return args_list


def _make_folder_name_from_options_and_dataset_name(options, dataset_name):
    cwd = Path(".").resolve()
    blocks = [
        f"sol_seed_{options['solution_seed']}",
        f"est_seed_{options['estimation_seed']}",
        f"tau_{options['estimation_tau']}",
        f"draws_est_{options['estimation_draws']}",
        f"draws_sol_{options['solution_draws']}",
        f"sequence_{options['monte_carlo_sequence']}",
        f"dataset_{dataset_name}",
    ]
    folder_name = "__".join(blocks)
    return cwd / folder_name


if __name__ == "__main__":

    data_dir = Path(".").resolve() / "old_kw94_one_pickles"
    dataset_names = [
        "data_5000_5000_emax_draws",
        "data_5000",
        "data_2000_5000_emax_draws",
        "data_2000",
        "data_1000_5000_emax_draws",
        "data_1000",
    ]

    dataset_dict = {}
    for dname in dataset_names:
        path = (data_dir / dname).with_suffix(".pickle")
        df = pd.read_pickle(path)
        dataset_dict[dname] = df

    start_params = pd.read_pickle("old_kw94_one_pickles/params.pickle")
    _, base_options = rp.get_example_model("kw_94_one", with_data=False)

    # calculate many scenarios with small dataset and not too many draws

    sequences = ["random", "sobol"]
    draws = [(3000, 2000), (500, 200), (1000, 400), (1500, 600), (2000, 800)]
    taus = [10, 20, 50, 100, 200, 500]
    # sequences = ["random"]
    # draws = [(200, 100)]
    # taus = [500]

    size = 51

    for dname, data in dataset_dict.items():
        for seq in sequences:
            for tau in taus:
                for sol_draws, est_draws in draws:
                    opt = base_options.copy()
                    opt["monte_carlo_sequence"] = seq
                    opt["estimation_tau"] = tau
                    opt["solution_draws"] = sol_draws
                    opt["estimation_draws"] = est_draws

                    arg_list = make_arg_list(start_params, opt, dname, size)
                    cf = get_data_collecting_crit_func(start_params, opt, data)

                    def my_cf(arg):
                        return cf(arg)

                    if "data_5000" in dname and est_draws > 1000:
                        n_processes = 1
                    elif "data_5000" in dname or est_draws > 1000:
                        n_processes = 2
                    elif "data_2000" in dname and est_draws >= 800:
                        n_processes = 4
                    elif "data_2000" in dname or est_draws >= 800:
                        n_processes = 6
                    else:
                        n_processes = 8

                    p = Pool(processes=n_processes)
                    p.map(my_cf, arg_list)
