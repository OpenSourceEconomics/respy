import copy

import numpy as np
import pandas as pd

from respy.likelihood import get_criterion_function
from respy.likelihood import get_parameter_vector
from respy.pre_processing.model_processing import _extract_cholesky
from respy.pre_processing.model_processing import parameters_to_dictionary
from respy.pre_processing.model_processing import parameters_to_vector
from respy.pre_processing.model_processing import process_model_spec
from respy.shared import cholesky_to_coeffs
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import minimal_simulate_observed


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
    attr = process_model_spec(params_spec, options_spec)
    df = minimal_simulate_observed(attr)

    # Evaluate at different points, ensuring that the simulated dataset still fits.
    params_spec, options_spec = generate_random_model(point_constr=constr)
    attr = process_model_spec(params_spec, options_spec)
    x = get_parameter_vector(attr)
    crit_func = get_criterion_function(attr, df)

    crit_func(x)


def test_invariant_results_for_two_estimations():
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation": {"agents": num_agents},
        "num_periods": np.random.randint(1, 4),
        "estimation": {"agents": num_agents},
    }

    # Simulate a dataset.
    params_spec, options_spec = generate_random_model(point_constr=constr)
    attr = process_model_spec(params_spec, options_spec)
    df = minimal_simulate_observed(attr)

    x = get_parameter_vector(attr)
    crit_func = get_criterion_function(attr, df)

    # First estimation.
    crit_val = crit_func(x)

    # Second estimation.
    crit_val_ = crit_func(x)

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
    attr = process_model_spec(params_spec, options_spec)
    df = minimal_simulate_observed(attr)

    base_val, edu_start_base = (None, np.random.randint(1, 5, size=1).tolist()[0])

    # We need to ensure that the initial lagged activity always has the same
    # distribution.
    edu_lagged_base = np.random.uniform(size=5).tolist()

    for num_edu_start in [1, np.random.choice([2, 3, 4]).tolist()]:

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

        attr = process_model_spec(params_spec, options_spec)
        df = minimal_simulate_observed(attr)

        x = get_parameter_vector(attr)
        crit_func = get_criterion_function(attr, df)

        val = crit_func(x)

        if base_val is None:
            base_val = val

        np.testing.assert_almost_equal(base_val, val)


def test_invariance_to_order_of_initial_schooling_levels():
    """Test invariance to order of initial schooling levels.

    This test ensures that the order of the initial schooling level specified in the
    model specification does not matter for the simulation of a dataset and subsequent
    evaluation of the criterion function.

    Warning
    -------
    This test fails if types have the identical intercept as no unique ordering is
    determined.

    """
    point_constr = {
        # We cannot allow for interpolation as the order of states within each
        # period changes and thus the prediction model is altered even if the same
        # state identifier is used.
        "interpolation": {"flag": False}
    }

    params_spec, options_spec = generate_random_model(point_constr=point_constr)

    attr = process_model_spec(params_spec, options_spec)

    edu_baseline_spec = attr["edu_spec"]
    num_types = attr["num_types"]
    num_paras = attr["num_paras"]
    optim_paras = attr["optim_paras"]

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

    # We are only looking at a single evaluation as otherwise the reordering affects
    # the optimizer that is trying better parameter values one-by-one. The
    # reordering might also violate the bounds.
    for i in range(53, num_paras):
        optim_paras["paras_bounds"][i] = [None, None]
        optim_paras["paras_fixed"][i] = False

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

    base_data, base_val = None, None

    for optim_paras in [optim_paras_baseline, optim_paras_shuffled]:
        for edu_spec in [edu_baseline_spec, edu_shuffled_spec]:

            attr["edu_spec"] = edu_spec

            # There is some more work to do to update the coefficients as we
            # distinguish between the economic and optimization version of the
            # parameters.
            x = parameters_to_vector(optim_paras, num_paras, "all", True)
            shocks_cholesky, _ = _extract_cholesky(x)
            shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)
            x[43:53] = shocks_coeffs
            attr["optim_paras"] = parameters_to_dictionary(
                paras_vec=x, is_debug=True, paras_type="econ"
            )

            df = minimal_simulate_observed(attr)

            if base_data is None:
                base_data = df.copy()

            pd.testing.assert_frame_equal(base_data, df)

            # This part checks the equality of a single function evaluation.
            x = get_parameter_vector(attr)
            crit_func = get_criterion_function(attr, df)
            val = crit_func(x)

            if base_val is None:
                base_val = val
            np.testing.assert_almost_equal(base_val, val)
