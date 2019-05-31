"""Test the simulation routine."""
import numpy as np
import pandas as pd
import pytest

import respy as rp
from respy.likelihood import get_crit_func
from respy.pre_processing.model_processing import process_model_spec
from respy.simulate import get_continuation_value_and_ex_post_rewards
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize(
    "seed",
    [
        0,
        pytest.param(1, marks=pytest.mark.xfail(reason="INADMISSIBILITY_PENALTY")),
        2,
        3,
        4,
        5,
        6,
        pytest.param(7, marks=pytest.mark.xfail(reason="INADMISSIBILITY_PENALTY")),
        8,
        pytest.param(9, marks=pytest.mark.xfail(reason="INADMISSIBILITY_PENALTY")),
    ],
)
def test_equality_of_total_values_and_rewexpost_for_myopic_individuals(seed):
    """Test equality of ex-post rewards and total values for myopic individuals."""
    np.random.seed(seed)

    # We need to simulate the model to get the emaxs and model attributes.
    params_spec, options_spec = generate_random_model(myopic=True)
    _, optim_paras = process_model_spec(params_spec, options_spec)

    draws = np.random.randn(1, 4)

    state_space, _ = rp.simulate(params_spec, options_spec)

    for period in range(state_space.num_periods):
        # Unpack necessary attributes
        rewards_period = state_space.get_attribute_from_period("rewards", period)
        emaxs_period = state_space.get_attribute_from_period("emaxs", period)[:, :4]
        max_education_period = (
            state_space.get_attribute_from_period("states", period)[:, 3]
            >= state_space.edu_max
        )

        total_values, rewards_ex_post = get_continuation_value_and_ex_post_rewards(
            rewards_period[:, -2:],
            rewards_period[:, :4],
            emaxs_period,
            draws,
            optim_paras["delta"],
            max_education_period,
        )

        np.testing.assert_equal(total_values, rewards_ex_post)


@pytest.mark.parametrize("seed", range(20))
def test_equality_for_myopic_agents_and_tiny_delta(seed):
    """Test equality of simulated data and likelihood myopic agents and tiny delta."""
    np.random.seed(seed)

    # Get simulated data and likelihood for myopic model.
    params_spec, options_spec = generate_random_model(myopic=True)

    state_space, df = rp.simulate(params_spec, options_spec)

    crit_func = get_crit_func(params_spec, options_spec, df)

    likelihood = crit_func(params_spec)

    # Get simulated data and likelihood for model with tiny delta.
    params_spec.loc["delta", "para"] = 1e-12

    state_space_, df_ = rp.simulate(params_spec, options_spec)

    crit_func_ = rp.get_crit_func(params_spec, options_spec, df_)

    likelihood_ = crit_func_(params_spec)

    pd.testing.assert_frame_equal(
        state_space.to_frame().drop(
            columns=["emax_a", "emax_b", "emax_edu", "emax_home", "emax"]
        ),
        state_space_.to_frame().drop(
            columns=["emax_a", "emax_b", "emax_edu", "emax_home", "emax"]
        ),
    )
    pd.testing.assert_frame_equal(df, df_)
    np.testing.assert_almost_equal(likelihood, likelihood_, decimal=15)
