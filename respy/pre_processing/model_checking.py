import numpy as np
import pandas as pd


def _validate_options(o):
    # Education.
    assert isinstance(o["education_lagged"], list)
    assert isinstance(o["education_start"], list)
    assert isinstance(o["education_share"], list)
    assert all(0 <= item <= 1 for item in o["education_lagged"])
    assert all(_is_positive_integer(i) for i in o["education_start"])
    assert all(0 <= item <= 1 for item in o["education_share"])
    assert all(i <= o["education_max"] for i in o["education_start"])
    np.testing.assert_almost_equal(np.sum(o["education_share"]), 1.0, decimal=4)
    assert _is_positive_integer(o["education_max"])

    # Estimation.
    assert _is_positive_integer(o["estimation_draws"]) and o["estimation_draws"] > 0
    assert _is_positive_integer(o["estimation_seed"])
    assert 0 <= o["estimation_tau"]

    # Interpolation.
    assert (
        _is_positive_integer(o["interpolation_points"])
        and o["interpolation_points"] != 0
    ) or o["interpolation_points"] == -1

    # Number of periods.
    assert _is_positive_integer(o["num_periods"])

    # Simulation.
    assert _is_positive_integer(o["simulation_agents"])
    assert _is_positive_integer(o["simulation_seed"])

    # Solution.
    assert _is_positive_integer(o["solution_draws"]) and o["solution_draws"] > 0
    assert _is_positive_integer(o["solution_seed"])

    # Covariates.
    assert isinstance(o["covariates"], dict)
    assert all(
        isinstance(key, str) and isinstance(val, str)
        for key, val in o["covariates"].items()
    )


def _is_positive_integer(x):
    return isinstance(x, (int, np.integer)) and x >= 0


def check_model_solution(options, optim_paras, state_space):
    # Distribute class attributes
    num_initial = len(options["education_start"])
    edu_start = options["education_start"]
    edu_start_max = max(edu_start)
    edu_max = options["education_max"]
    num_periods = options["num_periods"]
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
    assert np.all(np.isfinite(state_space.states))

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
    assert np.all(np.isfinite(state_space.emax))
