"""This module contains the functions for the generation of random requests."""
import json

import numpy as np

from respy.config import DATA_LABELS_EST
from respy.interface import minimal_simulation_interface
from respy.pre_processing.model_checking import check_model_attributes
from respy.pre_processing.model_processing import _create_attribute_dictionary
from respy.pre_processing.specification_helpers import csv_template


def generate_random_model(
    point_constr=None, bound_constr=None, num_types=None, file_path=None, myopic=False
):
    """Generate a random model specification.

    Parameters
    ----------
    point_constr : dict
        A full or partial options specification. Elements that are specified here are
        not drawn randomly.
    bound_constr : dict
        Upper bounds for some options to keep computation time reasonable. Can have the
        keys ["max_types", "max_periods", "max_edu_start", "max_agents", "max_draws"]
    num_types : int
        fix number of unobserved types.
    file_path : str
        save path for the output. The extensions .csv and .json are appended
        automatically.
    myopic : bool
        Indicator for myopic agents meaning the discount factor is set to zero.

    """
    # potential conversions from numpy._bool to python bool. Don't remove!
    myopic = bool(myopic)

    point_constr = {} if point_constr is None else point_constr
    bound_constr = {} if bound_constr is None else bound_constr

    for constr in point_constr, bound_constr:
        assert isinstance(constr, dict)

    bound_constr = _consolidate_bound_constraints(bound_constr)

    option_categories = [
        "edu_spec",
        "solution",
        "simulation",
        "estimation",
        "program",
        "interpolation",
    ]

    options = {cat: {} for cat in option_categories}

    if num_types is None:
        num_types = np.random.randint(1, bound_constr["max_types"] + 1)

    params = csv_template(num_types=num_types, initialize_coeffs=False)
    params["para"] = np.random.uniform(low=-0.05, high=0.05, size=len(params))

    params.loc["delta", "para"] = 1 - np.random.uniform() if myopic is False else 0.0

    shock_coeffs = params.loc["shocks"].index
    diagonal_shock_coeffs = [
        coeff for coeff in shock_coeffs if len(coeff.rsplit("_", 1)[1]) == 1
    ]
    for coeff in diagonal_shock_coeffs:
        params.loc[("shocks", coeff), "para"] = np.random.uniform(0.05, 1)

    gets_upper_bound = np.random.randint(0, 2, size=len(params)).astype(bool)
    nr_upper = gets_upper_bound.sum()
    params.loc[gets_upper_bound, "upper"] = params.loc[
        gets_upper_bound, "para"
    ] + np.random.uniform(0.1, 1, nr_upper)

    gets_lower_bound = np.random.randint(0, 2, size=len(params)).astype(bool)
    # don't replace already existing bounds
    gets_lower_bound = np.logical_and(gets_lower_bound, params["lower"].isnull())
    nr_lower = gets_lower_bound.sum()
    params.loc[gets_lower_bound, "lower"] = params.loc[
        gets_lower_bound, "para"
    ] - np.random.uniform(0.1, 1, nr_lower)

    params["fixed"] = np.random.choice([True, False], size=len(params), p=[0.1, 0.9])
    params.loc["shocks", "fixed"] = np.random.choice([True, False])
    if params["fixed"].to_numpy().all():
        params.loc["coeffs_a", "fixed"] = False

    params.loc["shocks", "fixed"] = False

    options["simulation"]["agents"] = np.random.randint(
        3, bound_constr["max_agents"] + 1
    )
    options["simulation"]["seed"] = np.random.randint(1, 1000)

    options["num_periods"] = np.random.randint(1, bound_constr["max_periods"])

    num_edu_start = np.random.randint(1, bound_constr["max_edu_start"] + 1)
    options["edu_spec"]["start"] = np.random.choice(
        np.arange(1, 15), size=num_edu_start, replace=False
    ).tolist()
    options["edu_spec"]["lagged"] = np.random.uniform(size=num_edu_start).tolist()
    options["edu_spec"]["share"] = get_valid_shares(num_edu_start)
    options["edu_spec"]["max"] = np.random.randint(
        max(options["edu_spec"]["start"]) + 1, 30
    )

    options["solution"]["draws"] = np.random.randint(1, bound_constr["max_draws"])
    options["solution"]["seed"] = np.random.randint(1, 10000)

    options["estimation"]["agents"] = np.random.randint(
        1, options["simulation"]["agents"]
    )
    options["estimation"]["draws"] = np.random.randint(1, bound_constr["max_draws"])
    options["estimation"]["seed"] = np.random.randint(1, 10000)

    options["estimation"]["tau"] = np.random.uniform(100, 500)

    options["program"]["debug"] = True

    # don't remove the seemingly redundant conversion from numpy._bool to python bool!
    options["interpolation"]["flag"] = bool(np.random.choice([True, False]))
    options["interpolation"]["points"] = np.random.randint(10, 100)

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
        params.to_csv(file_path + ".csv")

    return params, options


def _consolidate_bound_constraints(bound_constr):
    constr = {"max_types": 3, "max_periods": 3, "max_edu_start": 3}
    constr.update({"max_agents": 1000, "max_draws": 100})
    constr.update(bound_constr)
    return constr


def get_valid_shares(num_groups):
    """We simply need a valid request for the shares of types summing to one."""
    shares = np.random.np.random.uniform(size=num_groups)
    shares = shares / np.sum(shares)
    shares = shares.tolist()
    return shares


def minimal_simulate_observed(attr, is_missings=True):
    """ This function adds two important features of observed datasests: (1) missing
    observations and missing wage information.
    """

    def drop_agents_obs(agent):
        """ We now determine the exact period from which onward the history is truncated
        and cut the simulated dataset down to size.
        """
        start_truncation = np.random.np.random.choice(
            range(1, agent["Period"].max() + 2)
        )
        agent = agent[agent["Period"] < start_truncation]
        return agent

    state_space, df = minimal_simulation_interface(attr)

    np.random.seed(attr["seed_sim"])

    if is_missings:
        # We truncate the histories of agents. This mimics the frequent empirical fact
        # that we loose track of more and more agents over time.
        data_subset = df.groupby("Identifier").apply(drop_agents_obs)

        # We also want to drop the some wage observations. Note that we might be dealing
        # with a dataset where nobody is working anyway.
        is_working = data_subset["Choice"].isin([1, 2])
        num_drop_wages = int(
            np.sum(is_working) * np.random.np.random.uniform(high=0.5, size=1)
        )
        if num_drop_wages > 0:
            indices = data_subset["Wage"][is_working].index
            index_missing = np.random.np.random.choice(indices, num_drop_wages, False)
            data_subset.loc[index_missing, "Wage"] = None
        else:
            pass
    else:
        data_subset = df

    # We can restrict the information to observed entities only.
    data_subset = data_subset[DATA_LABELS_EST]

    # TODO: Check whether this is necessary.
    data_subset.Wage = data_subset.Wage.round(6)

    return data_subset
