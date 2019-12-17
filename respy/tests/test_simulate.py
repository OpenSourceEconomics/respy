"""Test the simulation routine."""
import numpy as np
import pandas as pd
import pytest

import respy as rp
from respy.likelihood import get_crit_func
from respy.pre_processing.specification_helpers import generate_obs_labels
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("seed", range(20))
def test_equality_for_myopic_agents_and_tiny_delta(seed):
    """Test equality of simulated data and likelihood with myopia and tiny delta."""
    np.random.seed(seed)

    # Get simulated data and likelihood for myopic model.
    params, options = generate_random_model(myopic=True)

    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)

    crit_func = get_crit_func(params, options, df)
    likelihood = crit_func(params)

    # Get simulated data and likelihood for model with tiny delta.
    params.loc["delta", "value"] = 1e-12

    df_ = simulate(params)

    crit_func_ = rp.get_crit_func(params, options, df_)
    likelihood_ = crit_func_(params)

    pd.testing.assert_frame_equal(df, df_)
    np.testing.assert_almost_equal(likelihood, likelihood_, decimal=12)


@pytest.mark.parametrize("seed", range(20))
def test_equality_of_models_with_and_without_observables(seed):
    """Test equality of models with and without observables.

    First, generate a model where the parameter values of observables is set to zero.
    The second model is obtained by assigning all observable indicators the value of the
    constant in the reward functions and set the constants to zero. The two models
    should be equivalent.

    """
    np.random.seed(seed)

    # Now specify a set of observables
    observables = [np.random.randint(2, 6)]
    point_constr = {"observables": observables}

    # Get simulated data and likelihood for myopic model.
    params, options = generate_random_model(myopic=True, point_constr=point_constr)

    # Get all reward values
    index_reward = [
        x for x in set(params.index.get_level_values(0)) if "nonpec" in x or "wage" in x
    ]

    # Get all indices that have
    obs_labels = generate_obs_labels(observables, index_reward)

    # Set these values to zero
    params.loc[obs_labels, "value"] = 0

    # Simulate the base model
    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)

    # Put two new values into the eq
    for x in obs_labels:
        params.loc[x, "value"] = params.loc[(x[0], "constant"), "value"]

    for x in index_reward:
        params.loc[(x, "constant"), "value"] = 0

    # Simulate the new model
    df_ = simulate(params)

    # test for equality
    pd.testing.assert_frame_equal(df_, df)


@pytest.mark.parametrize("seed", range(20))
def test_distribution_of_observables(seed):
    """Test that the distribution of observables matches the simulated distribution."""
    np.random.seed(seed)

    # Now specify a set of observables
    point_constr = {
        "observables": [np.random.randint(2, 6)],
        "simulation_agents": 1000,
        "n_periods": 1,
    }

    params, options = generate_random_model(point_constr=point_constr)

    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)

    # Check observable probabilities
    probs = df["Observable_0"].value_counts(normalize=True, sort=False)

    # Check proportions
    np.testing.assert_almost_equal(
        probs.to_numpy(),
        params.loc[
            params.index.get_level_values(0).str.contains("observable_observable_0"),
            "value",
        ].to_numpy(),
        decimal=1,
    )
