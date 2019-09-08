"""This module contains the functions for the generation of random requests."""
import collections

import numpy as np
from estimagic.optimization.utilities import cov_matrix_to_sdcorr_params
from estimagic.optimization.utilities import number_of_triangular_elements_to_dimension

from respy.config import DEFAULT_OPTIONS
from respy.pre_processing.model_processing import process_params_and_options
from respy.pre_processing.specification_helpers import csv_template
from respy.shared import generate_column_labels_estimation
from respy.simulate import get_simulate_func


_BASE_CORE_STATE_SPACE_FILTERS = [
    # In periods > 0, if agents accumulated experience only in one choice, lagged choice
    # cannot be different.
    "period > 0 and exp_{i} == period and lagged_choice_1 != '{i}'",
    # In periods > 0, if agents always accumulated experience, lagged choice cannot be
    # non-experience choice.
    "period > 0 and exp_a + exp_b + exp_edu == period and lagged_choice_1 == '{j}'",
    # In periods > 0, if agents accumulated no years of schooling, lagged choice cannot
    # be school.
    "period > 0 and lagged_choice_1 == 'edu' and exp_edu == 0",
    # If experience in choice 0 and 1 are zero, lagged choice cannot be this choice.
    "lagged_choice_1 == '{k}' and exp_{k} == 0",
    # In period 0, agents cannot choose occupation a or b.
    "period == 0 and lagged_choice_1 == '{k}'",
]
"""list: List of core state space filters.

.. deprecated::

    This variable must be removed if generate_random_model is rewritten such that
    functions for each replicable paper are written.

"""


_BASE_COVARIATES = {
    "not_exp_a_lagged": "exp_a > 0 and lagged_choice_1 != 'a'",
    "not_exp_b_lagged": "exp_b > 0 and lagged_choice_1 != 'b'",
    "work_a_lagged": "lagged_choice_1 == 'a'",
    "work_b_lagged": "lagged_choice_1 == 'b'",
    "edu_lagged": "lagged_choice_1 == 'edu'",
    "not_any_exp_a": "exp_a == 0",
    "not_any_exp_b": "exp_b == 0",
    "any_exp_a": "exp_a > 0",
    "any_exp_b": "exp_b > 0",
    "hs_graduate": "exp_edu >= 12",
    "co_graduate": "exp_edu >= 16",
    "is_return_not_high_school": "~edu_lagged and ~hs_graduate",
    "is_return_high_school": "~edu_lagged and hs_graduate",
    "is_minor": "period < 2",
    "is_young_adult": "2 <= period <= 4",
    "is_adult": "5 <= period",
    "constant": "1",
    "exp_a_square": "exp_a ** 2 / 100",
    "exp_b_square": "exp_b ** 2 / 100",
    "up_to_nine_years_edu": "exp_edu <= 9",
    "at_least_ten_years_edu": "exp_edu >= 10",
}
"""dict: Dictionary containing specification of covariates.

.. deprecated::

    This variable must be removed if generate_random_model is rewritten such that
    functions for each replicable paper are written.

"""


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

    options = {
        **DEFAULT_OPTIONS,
        **options,
        "core_state_space_filters": _BASE_CORE_STATE_SPACE_FILTERS,
        "covariates": _BASE_COVARIATES,
    }

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
    """Simulate a (truncated) dataset.

    The data can have two more properties. First, truncated history, second, missing
    wages.

    """
    _, _, options = process_params_and_options(params, options)

    simulate = get_simulate_func(params, options)
    df = simulate(params)

    np.random.seed(options["simulation_seed"])

    if is_missings:
        # Truncate the histories of agents. This mimics the effect of attrition.
        # Histories can be truncated after the first period or not at all. So, all
        # individuals have at least one observation.
        period_of_truncation = df.groupby("Identifier").Period.transform(
            lambda x: np.random.choice(x.max() + 1) + 1
        )
        data_subset = df.loc[df.Period.lt(period_of_truncation)].copy()

        # Add some missings to wage data.
        is_working = data_subset["Choice"].isin(options["choices_w_wage"])
        num_drop_wages = int(is_working.sum() * np.random.uniform(high=0.5))

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
