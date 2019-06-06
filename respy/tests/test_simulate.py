"""Test the simulation routine."""
import numpy as np
import pandas as pd
import pytest

import respy as rp
from respy.likelihood import get_crit_func
from respy.pre_processing.model_processing import process_params
from respy.simulate import calculate_value_functions_and_flow_utilities
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("seed", range(10))
def test_equality_of_total_values_and_rewexpost_for_myopic_individuals(seed):
    """Test equality of ex-post rewards and total values for myopic individuals."""
    np.random.seed(seed)

    # We need to simulate the model to get the emaxs and model attributes.
    params, options = generate_random_model(myopic=True)
    params, optim_paras = process_params(params)

    draws = np.random.randn(1, 4)

    state_space, _ = rp.simulate(params, options)

    for period in range(state_space.num_periods):
        # Unpack necessary attributes
        wages = state_space.get_attribute_from_period("wages", period)
        nonpec = state_space.get_attribute_from_period("nonpec", period)
        emaxs_period = state_space.get_attribute_from_period(
            "continuation_values", period
        )
        is_inadmissible = state_space.get_attribute_from_period(
            "is_inadmissible", period
        )

        total_values, rewards_ex_post = calculate_value_functions_and_flow_utilities(
            wages, nonpec, emaxs_period, draws, optim_paras["delta"], is_inadmissible
        )

        # Only states without maximum education are tested as the inadmissibility
        # penalty is applied to the total values of states with maximum education.
        states_in_period = state_space.get_attribute_from_period("states", period)
        idx = np.where(states_in_period[:, 3] != state_space.edu_max)

        np.testing.assert_equal(total_values[idx], rewards_ex_post[idx])


@pytest.mark.parametrize("seed", range(20))
def test_equality_for_myopic_agents_and_tiny_delta(seed):
    """Test equality of simulated data and likelihood myopic agents and tiny delta."""
    np.random.seed(seed)

    # Get simulated data and likelihood for myopic model.
    params, options = generate_random_model(myopic=True)

    state_space, df = rp.simulate(params, options)

    crit_func = get_crit_func(params, options, df)

    likelihood = crit_func(params)

    # Get simulated data and likelihood for model with tiny delta.
    params.loc["delta", "para"] = 1e-12

    state_space_, df_ = rp.simulate(params, options)

    crit_func_ = rp.get_crit_func(params, options, df_)

    likelihood_ = crit_func_(params)

    pd.testing.assert_frame_equal(df, df_)
    np.testing.assert_almost_equal(likelihood, likelihood_, decimal=12)
