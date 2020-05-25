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


def simulate_and_save(arg):
    arg = arg.copy()
    out_path = arg.pop("path")
    data = simulate(**arg)
    pd.to_pickle(data, out_path)




if __name__ == "__main__":
    bld = Path(".").resolve() / "bld"
    if not bld.exists():
        bld.mkdir()

    model = "kw_data_one"
    n_seeds = 100
    n_processes = 25

    start_params = get_params(model)


    args_list = []

    for sim_seed in range(n_seeds):
        sol_seed = sim_seed + 1000
        arg = {}
        arg["simulation_seed"] = sim_seed
        arg["solution_seed"] = sol_seed
        arg["model"] = model
        arg["params"] = start_params
        arg["path"] = bld / f"seed_sim_{sim_seed}__sol_seed__{sol_seed}.pickle"
        args_list.append(arg)

    p = Pool(processes=n_processes)
    p.map(simulate_and_save, args_list)

