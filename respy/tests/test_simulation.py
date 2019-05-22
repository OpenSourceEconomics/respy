"""Test the simulation routine."""
import numpy as np
import pandas as pd
import pytest

from respy.pre_processing.model_processing import process_model_spec
from respy.python.interface import minimal_estimation_interface
from respy.python.interface import minimal_simulation_interface
from respy.python.shared.shared_auxiliary import (
    get_continuation_value_and_ex_post_rewards,
)
from respy.tests.codes.random_model import generate_random_model


@pytest.mark.parametrize("seed", range(10))
def test_relationships_in_simulated_data(seed):
    """Test some relationships in simulated data.

    The following relations are tested:

    - Wages can be calculated

    """
    np.random.seed(seed)

    # Set model constraints
    is_deterministic = np.random.choice([True, False])
    is_myopic = np.random.choice([True, False])
    max_draws = np.random.randint(5, 200)
    bound_constr = {"max_draws": max_draws, "max_agents": max_draws}

    params_spec, options_spec = generate_random_model(
        bound_constr=bound_constr, deterministic=is_deterministic, myopic=is_myopic
    )
    attr = process_model_spec(params_spec, options_spec)

    state_space, df = minimal_simulation_interface(attr)

    # We can back out the wage information from other information provided in the
    # simulated dataset.
    for choice in [1, 2]:
        cond = df["Choice"] == choice
        label_sys = f"Systematic_Reward_{choice}"
        label_sho = f"Shock_Reward_{choice}"
        label_gen = f"General_Reward_{choice}"
        label_com = "Common_Reward"
        df["Ex_Post_Reward"] = (df[label_sys] - df[label_gen] - df[label_com]) * df[
            label_sho
        ]

        np.testing.assert_array_almost_equal(
            df["Ex_Post_Reward"].loc[cond], df["Wage"].loc[cond]
        )

    # In the myopic case, the total reward should the equal to the ex post rewards.
    if is_myopic:
        # The shock only affects the skill-function and not the other components
        # determining the overall reward.
        for choice in [1, 2]:
            cond = df["Choice"] == choice

            label = f"Ex_Post_Reward_{choice}"
            label_gen = f"General_Reward_{choice}"
            label_com = "Common_Reward"
            label_wag = "Wage"

            df[label] = df[label_wag] + df[label_gen] + df[label_com]

            col_1 = df[f"Total_Reward_{choice}"].loc[cond]
            col_2 = df[label].loc[cond]

            np.testing.assert_array_almost_equal(col_1, col_2)

        for choice in [3, 4]:
            label = f"Ex_Post_Reward_{choice}"
            label_sys = f"Systematic_Reward_{choice}"
            label_sho = f"Shock_Reward_{choice}"

            df[label] = df[label_sys] + df[label_sho]

            # The equality does not hold if a state is inadmissible.
            cond = df["Years_Schooling"] != state_space.edu_max

            col_1 = df["Total_Reward_" + str(choice)].loc[cond]
            col_2 = df[label].loc[cond]

            np.testing.assert_array_almost_equal(col_1, col_2)

    # If the model is deterministic, all shocks should be equal to zero. Of course,
    # one after exponentiation for wages.
    if is_deterministic:
        for i in range(1, 5):
            label = f"Shock_Reward_{i}"
            if i in [1, 2]:
                cond = df[label] == 1
            else:
                cond = df[label] == 0
            assert np.all(cond)


@pytest.mark.parametrize("seed", range(10))
def test_equality_of_total_values_and_rewexpost_for_myopic_individuals(seed):
    """Test equality of ex-post rewards and total values for myopic individuals."""
    np.random.seed(seed)

    # We need to simulate the model to get the emaxs and model attributes.
    params_spec, options_spec = generate_random_model(myopic=True)
    attr = process_model_spec(params_spec, options_spec)

    draws = np.random.randn(1, 4)

    state_space, _ = minimal_simulation_interface(attr)

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
            attr["optim_paras"]["delta"],
            max_education_period,
        )

        np.testing.assert_equal(total_values, rewards_ex_post)


@pytest.mark.parametrize("seed", range(10))
def test_equality_for_myopic_agents_and_tiny_delta(seed):
    """Test equality of simulated data and likelihood myopic agents and tiny delta."""
    np.random.seed(seed)

    # Get simulated data and likelihood for myopic model.
    params_spec, options_spec = generate_random_model(myopic=True)
    attr = process_model_spec(params_spec, options_spec)

    _, df = minimal_simulation_interface(attr)
    _, likelihood = minimal_estimation_interface(attr, df)

    # Get simulated data and likelihood for model with tiny delta.
    params_spec.loc["delta", "para"] = 1e-12
    attr = process_model_spec(params_spec, options_spec)

    _, df_ = minimal_simulation_interface(attr)
    _, likelihood_ = minimal_estimation_interface(attr, df_)

    pd.testing.assert_frame_equal(df, df_)
    assert likelihood == likelihood_
