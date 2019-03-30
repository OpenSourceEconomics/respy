"""Process model specification files or objects.

"""
import pandas as pd
import numpy as np
import json
import yaml
from respy.python.shared.shared_auxiliary import distribute_parameters, get_optim_paras
from respy.pre_processing.specification_helpers import csv_template
from pathlib import Path


def process_model_spec(params_spec, options_spec):
    if not isinstance(params_spec, pd.DataFrame):
        assert isinstance(
            params_spec, str
        ), "params_spec has to be a DataFrame or file path."
        params_spec = _read_params_spec(params_spec)
    if not isinstance(options_spec, dict):
        assert isinstance(
            options_spec, str
        ), "options_spec has to be a dictionary or file path."
        options_spec = _read_options_spec(options_spec)
    attr = _create_attribute_dictionary(params_spec, options_spec)
    return attr


def write_out_model_spec(attr, save_path):
    params_spec = _params_spec_from_attributes(attr)
    options_spec = _options_spec_from_attributes(attr)
    # todo: does this need index=False?
    params_spec.to_csv(save_path + '.csv')
    with open(save_path + '.json', 'w') as j:
        json.dump(options_spec, j)


def _options_spec_from_attributes(attr):
    estimation = {
        'file': attr['file_est'],
        'maxfun': attr['maxfun'],
        'agents': attr['num_agents_est'],
        'draws': attr['num_draws_prob'],
        'optimizer': attr['optimizer_used'],
        'seed': attr['seed_prob'],
        'tau': attr['tau']
    }

    simulation = {
        'file': attr['file_sim'],
        'agents': attr['num_agents_sim'],
        'seed': attr['seed_sim']

    }

    program = {
        'debug': attr['is_debug'],
        'procs': attr['num_procs'],
        'threads': attr['num_threads'],
        'version': attr['version']
    }

    interpolation = {
        'flag': attr['is_interpolated'],
        'points': attr['num_points_interp'],
    }

    solution = {
        'store': attr['is_store'],
        'seed': attr['seed_emax'],
        'draws': attr['num_draws_emax']
    }

    options_spec = {
        'estimation': estimation,
        'simulation': simulation,
        'program': program,
        'interpolation': interpolation,
        'solution': solution,
        'preconditioning': attr['precond_spec'],
        'derivatives': attr['derivatives'],
        'edu_spec': attr['edu_spec'],
        'num_periods': attr['num_periods'],
    }

    for optimizer, option in attr['optimizer_options'].items():
        options_spec[optimizer] = option

    return options_spec


def _params_spec_from_attributes(attr):
    csv = csv_template(attr['num_types'])
    bounds = np.array(attr['optim_paras']['paras_bounds'])
    csv['lower'] = bounds[:, 0]
    csv['upper'] = bounds[:, 1]
    csv['fixed'] = attr['optim_paras']['paras_fixed']
    csv['para'] = get_optim_paras(
        paras_dict=attr['optim_paras'],
        num_paras=attr['num_paras'],
        which='all',
        is_debug=True)
    return csv


def _create_attribute_dictionary(params_spec, options_spec):
    attr = {
        "edu_max": int(options_spec["edu_spec"]["max"]),
        "file_est": str(options_spec["estimation"]["file"]),
        "file_sim": str(options_spec["simulation"]["file"]),
        "is_debug": bool(options_spec["program"]["debug"]),
        "is_interpolated": bool(options_spec["interpolation"]["flag"]),
        "is_store": bool(options_spec["solution"]["store"]),
        "maxfun": int(options_spec["estimation"]["maxfun"]),
        "num_agents_est": int(options_spec["estimation"]["agents"]),
        "num_agents_sim": int(options_spec["simulation"]["agents"]),
        "num_draws_emax": int(options_spec["solution"]["draws"]),
        "num_draws_prob": int(options_spec["estimation"]["draws"]),
        "num_points_interp": int(options_spec["interpolation"]["points"]),
        "num_procs": int(options_spec["program"]["procs"]),
        "num_threads": int(options_spec["program"]["threads"]),
        "num_types": int(_get_num_types(params_spec)),
        "optim_paras": distribute_parameters(params_spec["para"].to_numpy(), is_debug=True),
        "optimizer_used": str(options_spec["estimation"]["optimizer"]),
        # make type conversions here
        "precond_spec": options_spec["preconditioning"],
        "seed_emax": int(options_spec["solution"]["seed"]),
        "seed_prob": int(options_spec["estimation"]["seed"]),
        "seed_sim": int(options_spec["simulation"]["seed"]),
        "tau": float(options_spec["estimation"]["tau"]),
        "version": str(options_spec["program"]["version"]),
        "derivatives": str(options_spec['derivatives']),
        # to-do: add type conversions and checks for edu spec
        'edu_spec': options_spec['edu_spec'],
        'num_periods': int(options_spec['num_periods']),
        'num_paras': len(params_spec),
    }

    # todo: add assert statements for bounds
    bounds = []
    for coeff in params_spec.index:
        bound = []
        for bounds_type in ['lower', 'upper']:
            if pd.isnull(params_spec.loc[coeff, bounds_type]):
                bound.append(None)
            else:
                bound.append(float(params_spec.loc[coeff, bounds_type]))
        bounds.append(bound)

    attr["optim_paras"]["paras_bounds"] = bounds
    attr["optim_paras"]["paras_fixed"] = params_spec["fixed"].astype(bool).to_numpy().tolist()

    optimizers = [
        "FORT-NEWUOA",
        "FORT-BFGS",
        "FORT-BOBYQA",
        "SCIPY-BFGS",
        "SCIPY-POWELL",
        "SCIPY-LBFGSB",
    ]
    # to-do: add type checks and/or conversions for optimizer options
    attr["optimizer_options"] = {}
    for opt in optimizers:
        attr["optimizer_options"][opt] = options_spec[opt]

    attr['is_myopic'] = params_spec.loc[('delta', 'delta'), 'para'] == 0.0

    # to-do: make asserts that all string values are lowercase
    return attr


def _get_num_types(params_spec):
    if 'type_shares' in params_spec.index:
        len_type_shares = len(params_spec.loc["type_shares"])
        num_types = len_type_shares / 2 + 1
    else:
        num_types = 1
    return num_types


def _read_params_spec(file_path):
    assert file_path.endswith(".csv"), "file_path has to be a .csv file"
    params_spec = pd.read_csv(file_path)
    params_spec["para"] = params_spec["para"].astype(float)
    params_spec.set_index(['category', 'name'], inplace=True)
    return params_spec


def _read_options_spec(file_path):
    if file_path.endswith(".json"):
        with open(file_path, "r") as j:
            options_spec = json.load(j)
    elif file_path.endswith(".yaml"):
        with open(file_path, "r") as y:
            options_spec = yaml.load(y)

    default = default_model_dict()
    default.update(options_spec)
    options_spec = default
    return options_spec


def default_model_dict():
    """Return a partial init_dict with default values.

    This is not a complete init_dict. It only contains the parts
    for which default values make sense.

    """
    default = {
        "FORT-NEWUOA": {"maxfun": 1000000, "npt": 1, "rhobeg": 1.0, "rhoend": 0.000001},
        "FORT-BFGS": {"eps": 0.0001, "gtol": 0.00001, "maxiter": 10, "stpmx": 100.0},
        "FORT-BOBYQA": {"maxfun": 1000000, "npt": 1, "rhobeg": 1.0, "rhoend": 0.000001},
        "SCIPY-BFGS": {"eps": 0.0001, "gtol": 0.0001, "maxiter": 1},
        "SCIPY-POWELL": {
            "ftol": 0.0001,
            "maxfun": 1000000,
            "maxiter": 1,
            "xtol": 0.0001,
        },
        "SCIPY-LBFGSB": {
            "eps": 0.000000441037423,
            "factr": 30.401091854739622,
            "m": 5,
            "maxiter": 2,
            "maxls": 2,
            "pgtol": 0.000086554171164,
        },
    }

    return default
