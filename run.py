from respy.estimagic_interface import simulate
from respy.estimagic_interface import get_params
from respy.estimagic_interface import collect_likelihood_data
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

def one_arg_collect_and_save_likelihood_data(arg):
    """Collect likelihood data and save the result as pickle file

    arg is a dict with the following entries:

    "params": pd.DataFrame
    "model": str
    "loc": tuple that selects the parameter to be modified
    "new_val": new value of the modified parameter
    "solution_seed": int
    "estimation_seed": int
    "data": pd.DataFrame

    """
    params = arg["params"].copy(deep=True)
    params.loc[arg["loc"], "value"] = arg["new_val"]
    try:
        like_data = collect_likelihood_data(
            params=params,
            data=arg["data"],
            model=arg["model"],
            solution_seed=arg["solution_seed"],
            estimation_seed=arg["estimation_seed"],
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        like_data = {}

    name = "{}__sol_seed_{}__est_seed_{}__loc_{}__new_val_{}.pickle"
    name = name.format(
        arg["model"],
        arg["solution_seed"],
        arg["estimation_seed"],
        arg["loc"],
        arg["new_val"],
    )
    path = Path(".").resolve() / "bld" / name
    pd.to_pickle(like_data, path)


def _get_arg_list(model, data, n_seeds, size=31):
    max_abs_deviation = 0.05
    max_rel_deviation = 0.05
    est_seeds = range(n_seeds)
    sol_seeds = [s + 1000 for s in est_seeds]
    params = get_params(model)
    multpliers = np.linspace(1 - max_rel_deviation, 1 + max_rel_deviation, size)
    abs_values = np.linspace(-max_abs_deviation, max_abs_deviation, size)

    arg_list = []

    for par in params.index:
        for est_seed, sol_seed in zip(est_seeds, sol_seeds):
            for m, a in zip(multpliers, abs_values):
                if params.loc[par, "value"] == 0:
                    new_val = a
                else:
                    new_val = m * params.loc[par, "value"]

                arg = {
                    "params": params,
                    "model": model,
                    "loc": par,
                    "new_val": new_val,
                    "solution_seed": sol_seed,
                    "estimation_seed": est_seed,
                    "data": data
                }
                arg_list.append(arg)
    return arg_list


if __name__ == "__main__":
    bld = Path(".").resolve() / "bld"
    if not bld.exists():
        bld.mkdir()

    models = ["kw_data_one"]
    n_seeds = 3
    size = 31
    n_processes = 24

    for model in models:
        start_params = get_params(model)
        data = simulate(start_params, model)
        arg_list = _get_arg_list(model, data, n_seeds, size)
        p = Pool(processes=n_processes)
        p.map(one_arg_collect_and_save_likelihood_data, arg_list)

