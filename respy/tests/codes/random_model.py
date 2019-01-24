""" This module contains the functions for the generation of random requests.
"""
import numpy as np
import json

from respy.pre_processing.model_processing import _create_attribute_dictionary
from respy.pre_processing.model_checking import check_model_attributes
from respy.pre_processing.specification_helpers import csv_template
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.tests.codes.auxiliary import OPTIMIZERS_EST
from respy.tests.codes.auxiliary import get_valid_shares
from numpy.random import randint, uniform, choice
import pickle


def generate_random_model(
    point_constr={}, bound_constr={}, num_types=None, file_path=None,
    deterministic=False, myopic=False
):
    """Generate a random model specification.

    Args:
        point_constr (dict): A full or partial options specification. Elements that
            are specified here are not drawn randomly.
        bound_constr (dict): Upper bounds for some options to keep computation time
            reasonable. Can have the keys ["max_types", "max_periods",
            "max_edu_start", "max_agents", "max_draws"]
        num_types (int, optional): fix number of unobserved types.
        file_path (str, optional): save path for the output. The extensions .csv and
        .json are appended automatically.

    """
    for constr in point_constr, bound_constr:
        assert isinstance(constr, dict)

    if "program" in point_constr and "version" in point_constr["program"]:
        version = point_constr["program"]["version"]
    else:
        version = choice(["python", "fortran"])

    bound_constr = _consolidate_bound_constraints(bound_constr, version)

    option_categories = [
        "edu_spec",
        "solution",
        "simulation",
        "estimation",
        "preconditioning",
        "program",
        "interpolation",
    ]

    options = {cat: {} for cat in option_categories}

    if num_types is None:
        num_types = randint(1, bound_constr["max_types"] + 1)

    params = csv_template(num_types=num_types, initialize_coeffs=False)
    params["para"] = uniform(low=-0.05, high=0.05, size=len(params))
    if myopic is False:
        params.loc["delta", "para"] = choice([0.0, uniform()])
    else:
        params.loc["delta", "para"] = 0.0

    shock_coeffs = params.loc["shocks"].index
    diagonal_shock_coeffs = [
        coeff for coeff in shock_coeffs if len(coeff.rsplit("_", 1)[1]) == 1
    ]
    for coeff in diagonal_shock_coeffs:
        params.loc[('shocks', coeff), "para"] = uniform(0.05, 1)

    gets_upper_bound = randint(0, 2, size=len(params)).astype(bool)
    nr_upper = gets_upper_bound.sum()
    params.loc[gets_upper_bound, "upper"] = params.loc[
        gets_upper_bound, "para"
    ] + uniform(0.1, 1, nr_upper)

    gets_lower_bound = randint(0, 2, size=len(params)).astype(bool)
    # don't replace already existing bounds
    gets_lower_bound = np.logical_and(gets_lower_bound, params["lower"].isnull())
    nr_lower = gets_lower_bound.sum()
    params.loc[gets_lower_bound, "lower"] = params.loc[
        gets_lower_bound, "para"
    ] - uniform(0.1, 1, nr_lower)

    params["fixed"] = choice(
        [True, False], size=len(params), p=[0.1, 0.9]
    )
    params.loc['shocks', 'fixed'] = choice([True, False])
    if params["fixed"].values.all():
        params.loc['coeffs_a', 'fixed'] = False
    if params.loc['shocks', 'fixed'].values.any() or deterministic is True:
        params.loc['shocks', 'para'] = 0.0

    options["simulation"]["agents"] = randint(3, bound_constr["max_agents"] + 1)
    options['simulation']['seed'] = randint(1, 1000)
    options['simulation']['file'] = 'data'

    options["num_periods"] = randint(1, bound_constr["max_periods"])

    num_edu_start = randint(1, bound_constr["max_edu_start"] + 1)
    options['edu_spec']["start"] = choice(np.arange(1, 15), size=num_edu_start, replace=False).tolist()
    options['edu_spec']["lagged"] = uniform(size=num_edu_start).tolist()
    options['edu_spec']["share"] = get_valid_shares(num_edu_start)
    options['edu_spec']["max"] = randint(max(options['edu_spec']["start"]) + 1, 30)


    options['solution']['draws'] = randint(1, bound_constr['max_draws'])
    options['solution']['seed'] = randint(1, 10000)
    # don't remove the seemingly redundant conversion from numpy._bool to python bool!
    options['solution']['store'] = bool(choice([True, False]))

    options['estimation']['agents'] = randint(1, options['simulation']['agents'])
    options['estimation']['draws'] = randint(1, bound_constr['max_draws'])
    options['estimation']['seed'] = randint(1, 10000)
    options['estimation']['file'] = 'data.respy.dat'
    if version == 'fortran':
        options['estimation']['optimizer'] = choice(OPT_EST_FORT)
    else:
        options['estimation']['optimizer'] = choice(OPT_EST_PYTH)
    options['estimation']['maxfun'] = randint(1, 1000)
    options['estimation']['tau'] = uniform(100, 500)

    options['derivatives'] = 'forward-differences'

    options['preconditioning']['minimum'] = uniform(0.0000001, 0.001)
    options['preconditioning']['type'] = choice(['gradient', 'identity', 'magnitudes'])
    options['preconditioning']['eps'] = uniform(0.0000001, 0.1)

    options['program']['version'] = version
    options['program']['debug'] = True
    options['program']['threads'] = 1
    options['program']['procs'] = 1

    if version == 'fortran':
        if IS_PARALLELISM_MPI is True:
            options['program']['procs'] = randint(1, 5)
        if IS_PARALLELISM_OMP is True:
            options['program']['threads'] = randint(1, 5)

    # don't remove the seemingly redundant conversion from numpy._bool to python bool!
    options['interpolation']['flag'] = bool(choice([True, False]))
    options['interpolation']['points'] = randint(10, 100)

    for optimizer in OPTIMIZERS_EST:
        options[optimizer] = generate_optimizer_options(optimizer, params)

    # todo: better error catching here to locate the problems.
    attr = _create_attribute_dictionary(params, options)
    check_model_attributes(attr)

    for key, value in point_constr.items():
        if isinstance(value, dict):
            options[key].update(value)
        else:
            options[key] = value

    attr = _create_attribute_dictionary(params, options)
    check_model_attributes(attr)

    if file_path is not None:
        with open(file_path + ".json", "w") as j:
            json.dump(options, j)
        # todo: does this need index=False?
        params.to_csv(file_path + ".csv")

    return params, options


def _consolidate_bound_constraints(bound_constr, version):
    if version == "fortran":
        constr = {"max_types": 4, "max_periods": 10, "max_edu_start": 4}
    else:
        constr = {"max_types": 3, "max_periods": 3, "max_edu_start": 3}
    constr.update({"max_agents": 1000, "max_draws": 100})
    constr.update(bound_constr)
    return constr


def generate_optimizer_options(which, params_spec):

    free_params = len(params_spec) - params_spec['fixed'].sum()
    dict_ = dict()

    if which == "SCIPY-BFGS":
        dict_["gtol"] = uniform(0.0000001, 0.1)
        dict_["maxiter"] = randint(1, 10)
        dict_["eps"] = uniform(1e-9, 1e-6)

    elif which == "SCIPY-LBFGSB":
        dict_["factr"] = uniform(10, 100)
        dict_["pgtol"] = uniform(1e-6, 1e-4)
        dict_["maxiter"] = randint(1, 10)
        dict_["maxls"] = randint(1, 10)
        dict_["m"] = randint(1, 10)
        dict_["eps"] = uniform(1e-9, 1e-6)

    elif which == "SCIPY-POWELL":
        dict_["xtol"] = uniform(0.0000001, 0.1)
        dict_["ftol"] = uniform(0.0000001, 0.1)
        dict_["maxfun"] = randint(1, 100)
        dict_["maxiter"] = randint(1, 100)

    elif which in ["FORT-NEWUOA", "FORT-BOBYQA"]:
        rhobeg = uniform(0.0000001, 0.001)
        dict_["maxfun"] = randint(1, 100)
        dict_["rhobeg"] = rhobeg
        dict_["rhoend"] = uniform(0.01, 0.99) * rhobeg

        # It is not recommended that N is larger than upper as the code might
        # break down due to a segmentation fault. See the source files for the
        # absolute upper bounds.
        lower = (free_params) + 2
        upper = 2 * (free_params) + 1
        dict_["npt"] = randint(lower, upper + 1)

    elif which == "FORT-BFGS":
        dict_["maxiter"] = randint(1, 100)
        dict_["stpmx"] = uniform(75, 125)
        dict_["gtol"] = uniform(0.0001, 0.1)
        dict_["eps"] = uniform(1e-9, 1e-6)

    else:
        raise NotImplementedError("The optimizer you requested is not implemented.")

    return dict_
