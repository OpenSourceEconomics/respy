import numpy as np
import pandas as pd


def check_model_attributes(a):
    # Debug status
    assert a["is_debug"] in [True, False]

    # Seeds
    for seed in [a["seed_sol"], a["seed_sim"], a["seed_est"]]:
        assert np.isfinite(seed)
        assert isinstance(seed, int)
        assert seed > 0

    # Number of simulated agents.
    assert np.isfinite(a["num_agents_sim"])
    assert isinstance(a["num_agents_sim"], int)
    assert a["num_agents_sim"] > 0

    # Number of periods.
    assert np.isfinite(a["num_periods"])
    assert isinstance(a["num_periods"], int)
    assert a["num_periods"] > 0

    # Number of draws for Monte Carlo integration
    assert np.isfinite(a["num_draws_sol"])
    assert isinstance(a["num_draws_sol"], int)
    assert a["num_draws_sol"] >= 0

    # Debugging mode
    assert a["is_debug"] in [True, False]

    # Window for smoothing parameter
    assert isinstance(a["tau"], float)
    assert a["tau"] > 0

    # Interpolation
    assert a["interpolation"] in [True, False]
    assert isinstance(a["num_points_interp"], int)
    assert a["num_points_interp"] > 0

    # Simulation of S-ML
    assert isinstance(a["num_draws_est"], int)
    assert a["num_draws_est"] > 0

    # Education
    assert isinstance(a["edu_spec"]["max"], int)
    assert a["edu_spec"]["max"] > 0
    assert isinstance(a["edu_spec"]["start"], list)
    assert len(a["edu_spec"]["start"]) == len(set(a["edu_spec"]["start"]))
    assert all(isinstance(item, int) for item in a["edu_spec"]["start"])
    assert all(item > 0 for item in a["edu_spec"]["start"])
    assert all(item <= a["edu_spec"]["max"] for item in a["edu_spec"]["start"])
    assert all(isinstance(item, float) for item in a["edu_spec"]["share"])
    assert all(0 <= item <= 1 for item in a["edu_spec"]["lagged"])
    assert all(0 <= item <= 1 for item in a["edu_spec"]["share"])
    np.testing.assert_almost_equal(np.sum(a["edu_spec"]["share"]), 1.0, decimal=4)


def check_model_solution(attr, optim_paras, state_space):
    # Distribute class attributes
    num_initial = len(attr["edu_spec"]["start"])
    edu_start = attr["edu_spec"]["start"]
    edu_start_max = max(edu_start)
    edu_max = attr["edu_spec"]["max"]
    num_periods = attr["num_periods"]
    num_types = optim_paras["num_types"]

    # Check period.
    assert np.all(np.isin(state_space.states[:, 0], range(num_periods)))

    # The sum of years of experiences cannot be larger than constraint time.
    assert np.all(
        state_space.states[:, 1:4].sum(axis=1)
        <= (state_space.states[:, 0] + edu_start_max)
    )

    # Sector experience cannot exceed the time frame.
    assert np.all(state_space.states[:, 1] <= num_periods)
    assert np.all(state_space.states[:, 2] <= num_periods)

    # The maximum of education years is never larger than ``edu_max``.
    assert np.all(state_space.states[:, 3] <= edu_max)

    # Lagged choices are always between one and four.
    assert np.isin(state_space.states[:, 4], [1, 2, 3, 4]).all()

    # States and covariates have finite and nonnegative values.
    assert np.all(state_space.states >= 0)
    assert np.all(state_space.base_covariates >= 0)
    assert np.all(np.isfinite(state_space.states))
    assert np.all(np.isfinite(state_space.base_covariates))

    # Check for duplicate rows in each period. We only have possible duplicates if there
    # are multiple initial conditions.
    assert not pd.DataFrame(state_space.states).duplicated().any()

    # Check the number of states in the first time period.
    num_states_start = num_types * num_initial * 2
    assert (
        state_space.get_attribute_from_period("states", 0).shape[0] == num_states_start
    )
    assert np.sum(state_space.indexer[0] >= 0) == num_states_start

    # Check that we have as many indices as states.
    assert state_space.states.shape[0] == (state_space.indexer >= 0).sum()

    # Check finiteness of rewards and emaxs.
    assert np.all(np.isfinite(state_space.wages))
    assert np.all(np.isfinite(state_space.nonpec))
    assert np.all(np.isfinite(state_space.emaxs))
