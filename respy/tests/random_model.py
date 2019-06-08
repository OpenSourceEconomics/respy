"""This module contains the functions for the generation of random requests."""
import numpy as np

from respy.config import DATA_LABELS_EST
from respy.config import DEFAULT_OPTIONS
from respy.pre_processing.model_checking import _validate_options
from respy.pre_processing.model_processing import cov_matrix_to_sdcorr_params
from respy.pre_processing.model_processing import (
    number_of_triangular_elements_to_dimension,
)
from respy.pre_processing.specification_helpers import csv_template
from respy.simulate import simulate


def generate_random_model(
    point_constr=None, bound_constr=None, num_types=None, myopic=False
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

    if num_types is None:
        num_types = np.random.randint(1, bound_constr["max_types"] + 1)

    params = csv_template(num_types=num_types, initialize_coeffs=False)
    params["para"] = np.random.uniform(low=-0.05, high=0.05, size=len(params))

    params.loc["delta", "para"] = 1 - np.random.uniform() if myopic is False else 0.0

    num_shock_coeffs = len(params.loc["shocks"])
    dim = number_of_triangular_elements_to_dimension(num_shock_coeffs)
    helper = np.eye(dim) * 0.5
    helper[np.tril_indices(dim, k=-1)] = np.random.uniform(
        -0.05, 0.2, size=(num_shock_coeffs - dim)
    )
    cov = helper.dot(helper.T)
    params.loc["shocks", "para"] = cov_matrix_to_sdcorr_params(cov)

    params.loc["meas_error", "para"] = np.random.uniform(
        low=0.001, high=0.1, size=len(params.loc["meas_error"])
    )

    options = {}

    options["simulation_agents"] = np.random.randint(3, bound_constr["max_agents"] + 1)
    options["simulation_seed"] = np.random.randint(1, 1000)

    options["num_periods"] = np.random.randint(1, bound_constr["max_periods"])

    num_edu_start = np.random.randint(1, bound_constr["max_edu_start"] + 1)
    options["education_start"] = np.random.choice(
        np.arange(1, 15), size=num_edu_start, replace=False
    ).tolist()
    options["education_lagged"] = np.random.uniform(size=num_edu_start).tolist()
    options["education_share"] = get_valid_shares(num_edu_start)
    options["education_max"] = np.random.randint(
        max(options["education_start"]) + 1, 30
    )

    options["solution_draws"] = np.random.randint(1, bound_constr["max_draws"])
    options["solution_seed"] = np.random.randint(1, 10000)

    options["estimation_draws"] = np.random.randint(1, bound_constr["max_draws"])
    options["estimation_seed"] = np.random.randint(1, 10000)
    options["estimation_tau"] = np.random.uniform(100, 500)

    # don't remove the seemingly redundant conversion from numpy._bool to python bool!
    not_interpolation = np.random.choice([True, False])
    options["interpolation_points"] = (
        -1 if not_interpolation else np.random.randint(10, 100)
    )

    options = {**DEFAULT_OPTIONS, **options}

    _validate_options(options)

    for key, value in point_constr.items():
        if isinstance(value, dict):
            options[key].update(value)
        else:
            options[key] = value

    _validate_options(options)

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


def simulate_truncated_data(params, options, is_missings=True):
    """Simulate a dataset.

    The data can have two more properties. First, truncated history, second, missing
    wages.

    """

    def drop_agents_obs(agent):
        """ We now determine the exact period from which onward the history is truncated
        and cut the simulated dataset down to size.
        """
        start_truncation = np.random.choice(range(1, agent["Period"].max() + 2))
        agent = agent[agent["Period"] < start_truncation]
        return agent

    state_space, df = simulate(params, options)

    np.random.seed(options["simulation_seed"])

    if is_missings:
        # We truncate the histories of agents. This mimics the frequent empirical fact
        # that we loose track of more and more agents over time.
        data_subset = (
            df.groupby("Identifier").apply(drop_agents_obs).reset_index(drop=True)
        )

        # We also want to drop the some wage observations. Note that we might be dealing
        # with a dataset where nobody is working anyway.
        is_working = data_subset["Choice"].isin([0, 1])
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

    return data_subset
