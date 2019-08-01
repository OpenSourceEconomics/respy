"""This module contains the functions for the generation of random requests."""
import collections

import numpy as np
import pandas as pd
from estimagic.optimization.utilities import cov_matrix_to_sdcorr_params
from estimagic.optimization.utilities import number_of_triangular_elements_to_dimension
from packaging import version

from respy.config import DEFAULT_OPTIONS
from respy.pre_processing.model_processing import process_params_and_options
from respy.pre_processing.specification_helpers import csv_template
from respy.shared import generate_column_labels_estimation
from respy.simulate import get_simulate_func


def generate_random_model(
    point_constr=None,
    bound_constr=None,
    n_types=None,
    n_type_covariates=None,
    myopic=False,
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
    n_types : int
        Number of unobserved types.
    n_type_covariates :
        Number of covariates to calculate type probabilities.
    myopic : bool
        Indicator for myopic agents meaning the discount factor is set to zero.

    """
    point_constr = {} if point_constr is None else point_constr
    bound_constr = {} if bound_constr is None else bound_constr

    for constr in point_constr, bound_constr:
        assert isinstance(constr, dict)

    bound_constr = _consolidate_bound_constraints(bound_constr)

    if n_types is None:
        n_types = np.random.randint(1, bound_constr["max_types"] + 1)
    if n_type_covariates is None:
        n_type_covariates = np.random.randint(2, 4)

    params = csv_template(
        n_types=n_types, n_type_covariates=n_type_covariates, initialize_coeffs=False
    )
    params["value"] = np.random.uniform(low=-0.05, high=0.05, size=len(params))

    params.loc["delta", "value"] = 1 - np.random.uniform() if myopic is False else 0.0

    n_shock_coeffs = len(params.loc["shocks"])
    dim = number_of_triangular_elements_to_dimension(n_shock_coeffs)
    helper = np.eye(dim) * 0.5
    helper[np.tril_indices(dim, k=-1)] = np.random.uniform(
        -0.05, 0.2, size=(n_shock_coeffs - dim)
    )
    cov = helper.dot(helper.T)
    params.loc["shocks", "value"] = cov_matrix_to_sdcorr_params(cov)

    params.loc["meas_error", "value"] = np.random.uniform(
        low=0.001, high=0.1, size=len(params.loc["meas_error"])
    )

    options = {
        "simulation_agents": np.random.randint(3, bound_constr["max_agents"] + 1),
        "simulation_seed": np.random.randint(1, 1000),
        "n_periods": np.random.randint(1, bound_constr["max_periods"]),
        "choices": {"edu": {}},
    }

    n_edu_start = np.random.randint(1, bound_constr["max_edu_start"] + 1)
    options["choices"]["edu"]["start"] = np.random.choice(
        np.arange(1, 15), size=n_edu_start, replace=False
    )
    options["choices"]["edu"]["lagged"] = np.random.uniform(size=n_edu_start)
    options["choices"]["edu"]["share"] = _get_initial_shares(n_edu_start)
    options["choices"]["edu"]["max"] = np.random.randint(
        max(options["choices"]["edu"]["start"]) + 1, 30
    )

    options["solution_draws"] = np.random.randint(1, bound_constr["max_draws"])
    options["solution_seed"] = np.random.randint(1, 10000)

    options["estimation_draws"] = np.random.randint(1, bound_constr["max_draws"])
    options["estimation_seed"] = np.random.randint(1, 10000)
    options["estimation_tau"] = np.random.uniform(100, 500)

    options["interpolation_points"] = -1

    options = {**DEFAULT_OPTIONS, **options}

    options = _update_nested_dictionary(options, point_constr)

    return params, options


def _consolidate_bound_constraints(bound_constr):
    constr = {
        "max_types": 3,
        "max_periods": 3,
        "max_edu_start": 3,
        "max_agents": 1000,
        "max_draws": 100,
    }
    constr.update(bound_constr)

    return constr


def _get_initial_shares(num_groups):
    """We simply need a valid request for the shares of types summing to one."""
    shares = np.random.uniform(size=num_groups)
    shares = shares / shares.sum()

    return shares


def simulate_truncated_data(params, options, is_missings=True):
    """Simulate a dataset.

    The data can have two more properties. First, truncated history, second, missing
    wages.

    """

    def drop_agents_obs(agent):
        """ We now determine the exact period from which onward the history is truncated
        and cut the simulated dataset down to size.
        """
        # For more details on this hacky solution see
        # https://github.com/OpenSourceEconomics/respy/pull/225#issuecomment-517254853.
        if (
            version.parse(pd.__version__) >= version.parse("0.25.0")
            and agent.index[0] == 0
        ):
            _ = np.random.choice(range(1, agent["Period"].max() + 2))

        start_truncation = np.random.choice(range(1, agent["Period"].max() + 2))
        agent = agent[agent["Period"].lt(start_truncation)]

        return agent

    _, _, options = process_params_and_options(params, options)

    simulate = get_simulate_func(params, options)
    df = simulate(params)

    np.random.seed(options["simulation_seed"])

    if is_missings:
        # We truncate the histories of agents. This mimics the frequent empirical fact
        # that we loose track of more and more agents over time.
        data_subset = (
            df.groupby("Identifier").apply(drop_agents_obs).reset_index(drop=True)
        )

        # We also want to drop the some wage observations. Note that we might be dealing
        # with a dataset where nobody is working anyway.
        is_working = data_subset["Choice"].isin(options["choices_w_wage"])
        num_drop_wages = int(np.sum(is_working) * np.random.uniform(high=0.5))

        if num_drop_wages > 0:
            indices = data_subset["Wage"][is_working].index
            index_missing = np.random.choice(indices, num_drop_wages, replace=False)

            data_subset.loc[index_missing, "Wage"] = np.nan
        else:
            pass
    else:
        data_subset = df

    # We can restrict the information to observed entities only.
    labels, _ = generate_column_labels_estimation(options)
    data_subset = data_subset[labels]

    return data_subset


def _update_nested_dictionary(dict_, other):
    """Update a nested dictionary with another dictionary.

    The basic ``.update()`` method of dictionaries adds non-existing keys or replaces
    existing keys which works fine for unnested dictionaries. For nested dictionaries,
    levels under the current level are not updated but overwritten. This function
    recursively loops over keys and values and inserts the value if it is not a
    dictionary. If it is a dictionary, it applies the same process again.

    """
    for key, value in other.items():
        if isinstance(value, collections.Mapping):
            dict_[key] = _update_nested_dictionary(dict_.get(key, {}), value)
        else:
            dict_[key] = value
    return dict_
