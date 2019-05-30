import copy

import numpy as np
import pandas as pd
import pytest

from respy.likelihood import get_crit_func
from respy.pre_processing.model_processing import _extract_cholesky
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.pre_processing.model_processing import parse_parameters
from respy.pre_processing.model_processing import process_model_spec
from respy.pre_processing.model_processing import stack_parameters
from respy.shared import cholesky_to_coeffs
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data


def test_simulation_and_estimation_with_different_models():
    """Test the evaluation of the criterion function not at the true parameters."""
    # Set constraints.
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation": {"agents": num_agents},
        "num_periods": np.random.randint(1, 4),
        "edu_spec": {"start": [7], "max": 15, "share": [1.0]},
        "estimation": {"agents": num_agents},
    }

    # Simulate a dataset
    params_spec, options_spec = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params_spec, options_spec)

    # Evaluate at different points, ensuring that the simulated dataset still fits.
    params_spec, options_spec = generate_random_model(point_constr=constr)
    crit_func = get_crit_func(params_spec, options_spec, df)

    crit_func(params_spec)


def test_invariant_results_for_two_estimations():
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation": {"agents": num_agents},
        "num_periods": np.random.randint(1, 4),
        "estimation": {"agents": num_agents},
    }

    # Simulate a dataset.
    params_spec, options_spec = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params_spec, options_spec)

    crit_func = get_crit_func(params_spec, options_spec, df)

    # First estimation.
    crit_val = crit_func(params_spec)

    # Second estimation.
    crit_val_ = crit_func(params_spec)

    assert crit_val == crit_val_


def test_invariance_to_initial_conditions():
    """Test invariance to initial conditions.

    We ensure that the number of initial conditions does not matter for the evaluation
    of the criterion function if a weight of one is put on the first group.

    """
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation": {"agents": num_agents},
        "num_periods": np.random.randint(1, 4),
        "edu_spec": {"max": np.random.randint(15, 25, size=1).tolist()[0]},
        "estimation": {"agents": num_agents},
        "interpolation": {"flag": False},
    }

    params_spec, options_spec = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params_spec, options_spec)

    edu_start_base = np.random.randint(1, 5, size=1).tolist()[0]

    # We need to ensure that the initial lagged activity always has the same
    # distribution.
    edu_lagged_base = np.random.uniform(size=5).tolist()

    likelihoods = []

    for num_edu_start in [1, 2, 3, 4]:

        # We always need to ensure that a weight of one is on the first level of
        # initial schooling.
        options_spec["edu_spec"]["share"] = [1.0] + [0.0] * (num_edu_start - 1)
        options_spec["edu_spec"]["lagged"] = edu_lagged_base[:num_edu_start]

        # We need to make sure that the baseline level of initial schooling is
        # always included. At the same time we cannot have any duplicates.
        edu_start = np.random.choice(
            range(1, 10), size=num_edu_start, replace=False
        ).tolist()
        if edu_start_base in edu_start:
            edu_start.remove(edu_start_base)
            edu_start.insert(0, edu_start_base)
        else:
            edu_start[0] = edu_start_base

        options_spec["edu_spec"]["start"] = edu_start

        df = simulate_truncated_data(params_spec, options_spec)

        crit_func = get_crit_func(params_spec, options_spec, df)

        likelihood = crit_func(params_spec)

        likelihoods.append(likelihood)

    assert np.equal.reduce(likelihoods)


@pytest.mark.parametrize(
    "seed",
    [
        0,
        1,
        2,
        3,
        pytest.param(
            4,
            marks=pytest.mark.xfail(
                reason="More than two types are not correctly sorted."
            ),
        ),
        5,
        6,
        7,
        8,
        9,
    ],
)
@pytest.mark.xfail
def test_invariance_to_order_of_initial_schooling_levels(seed):
    """Test invariance to order of initial schooling levels.

    This test ensures that the order of the initial schooling level specified in the
    model specification does not matter for the simulation of a dataset and subsequent
    evaluation of the criterion function.

    Warning
    -------
    This test fails if types have the identical intercept as no unique ordering is
    determined.

    """
    np.random.seed(seed)

    params_spec, options_spec = generate_random_model()
    attr, optim_paras = process_model_spec(params_spec, options_spec)

    edu_baseline_spec = attr["edu_spec"]
    num_types = optim_paras["num_types"]

    # We want to randomly shuffle the list of initial schooling but need to maintain
    # the order of the shares.
    shuffled_indices = np.random.permutation(range(len(edu_baseline_spec["start"])))
    edu_shuffled_start = np.array(edu_baseline_spec["start"])[shuffled_indices].tolist()
    edu_shuffled_lagged = np.array(edu_baseline_spec["lagged"])[
        shuffled_indices
    ].tolist()
    edu_shuffled_share = np.array(edu_baseline_spec["share"])[shuffled_indices].tolist()

    edu_shuffled_spec = copy.deepcopy(edu_baseline_spec)
    edu_shuffled_spec["lagged"] = edu_shuffled_lagged
    edu_shuffled_spec["start"] = edu_shuffled_start
    edu_shuffled_spec["share"] = edu_shuffled_share

    # We need to ensure that the baseline type is still in the first position.
    types_order = [0] + np.random.permutation(range(1, num_types)).tolist()

    type_shares = []
    for i in range(num_types):
        lower, upper = i * 2, (i + 1) * 2
        type_shares += [optim_paras["type_shares"][lower:upper].tolist()]

    optim_paras_baseline = copy.deepcopy(optim_paras)
    optim_paras_shuffled = copy.deepcopy(optim_paras)

    list_ = [optim_paras["type_shifts"][i, :].tolist() for i in types_order]
    optim_paras_shuffled["type_shifts"] = np.array(list_)

    list_ = [type_shares[i] for i in types_order]
    optim_paras_shuffled["type_shares"] = np.array(list_).flatten()

    base_df, base_likelihood = None, None

    for optim_paras in [optim_paras_baseline, optim_paras_shuffled]:
        for edu_spec in [edu_baseline_spec, edu_shuffled_spec]:

            options_spec["edu_spec"] = edu_spec

            # There is some more work to do to update the coefficients as we distinguish
            # between the economic and optimization version of the parameters.
            x = stack_parameters(optim_paras)
            shocks_cholesky = _extract_cholesky(x)
            shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)
            x[43:53] = shocks_coeffs
            optim_paras = parse_parameters(x, paras_type="econ")

            params_spec = _params_spec_from_attributes(optim_paras)
            options_spec = _options_spec_from_attributes(attr)

            df = simulate_truncated_data(params_spec, options_spec)

            if base_df is None:
                base_df = df.copy()

            pd.testing.assert_frame_equal(base_df, df)

            # This part checks the equality of a single function evaluation.
            crit_func = get_crit_func(params_spec, options_spec, df)
            likelihood = crit_func(params_spec)

            if base_likelihood is None:
                base_likelihood = likelihood
            np.testing.assert_almost_equal(base_likelihood, likelihood)
