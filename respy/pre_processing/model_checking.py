import numpy as np
import pandas as pd

from respy.config import PRINT_FLOAT


def check_model_attributes(a):
    # Number of parameters
    assert isinstance(a["num_paras"], int)
    assert a["num_paras"] >= 53

    # Debug status
    assert a["is_debug"] in [True, False]

    # Forward-looking agents
    assert a["myopia"] in [True, False]

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
    assert np.isfinite(a["num_draws_emax"])
    assert isinstance(a["num_draws_emax"], int)
    assert a["num_draws_emax"] >= 0

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
    assert isinstance(a["num_draws_prob"], int)
    assert a["num_draws_prob"] > 0

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

    # Check model parameters
    check_model_parameters(a["optim_paras"])

    # It is not clear at this point how to impose parameter constraints on
    # the covariance matrix in a flexible manner. So, either all fixed or
    # none. As a special case, we also allow for all off-diagonal elements
    # to be fixed to zero.
    shocks_coeffs = a["optim_paras"]["shocks_cholesky"][np.tril_indices(4)]
    shocks_fixed = np.array(a["optim_paras"]["paras_fixed"][43:53])

    all_free = not shocks_fixed.any()

    dim = len(a["optim_paras"]["shocks_cholesky"])
    helper = np.zeros((dim, dim))
    helper[np.tril_indices(dim)] = shocks_coeffs
    off_diagonals_zero = np.diag(helper).sum() == helper.sum()

    helper = np.zeros((dim, dim), dtype=bool)
    helper[np.tril_indices(dim)] = shocks_fixed
    off_diagonals_fixed = (helper[np.tril_indices(dim, k=-1)]).all()

    diagonal_matrix = off_diagonals_zero & off_diagonals_fixed

    if not (all_free or shocks_fixed.all() or diagonal_matrix):
        raise ValueError(" Misspecified constraints for covariance matrix")

    # Discount rate and type shares need to be larger than on at all times.
    for label in ["paras_fixed", "paras_bounds"]:
        assert isinstance(a["optim_paras"][label], list)
        assert len(a["optim_paras"][label]) == a["num_paras"]

    for i in range(1):
        assert a["optim_paras"]["paras_bounds"][i][0] >= 0.00


def check_model_solution(attr, state_space):
    # Distribute class attributes
    num_initial = len(attr["edu_spec"]["start"])
    edu_start = attr["edu_spec"]["start"]
    edu_start_max = max(edu_start)
    edu_max = attr["edu_spec"]["max"]
    num_periods = attr["num_periods"]
    num_types = attr["num_types"]

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
    assert np.all(state_space.covariates >= 0)
    assert np.all(np.isfinite(state_space.states))
    assert np.all(np.isfinite(state_space.covariates))

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
    assert np.all(np.isfinite(state_space.rewards))
    assert np.all(np.isfinite(state_space.emaxs))


def check_model_parameters(optim_paras):
    """Check the integrity of all model parameters."""
    # Auxiliary objects
    num_types = len(optim_paras["type_shifts"])

    # Checks for all arguments
    keys = [
        "coeffs_a",
        "coeffs_b",
        "coeffs_edu",
        "coeffs_home",
        "shocks_cholesky",
        "delta",
        "type_shares",
        "type_shifts",
        "coeffs_common",
    ]

    for key in keys:
        assert isinstance(optim_paras[key], np.ndarray), key
        assert np.all(np.isfinite(optim_paras[key]))
        assert optim_paras[key].dtype == "float"
        assert np.all(abs(optim_paras[key]) < PRINT_FLOAT)

    # Check for discount rate
    assert optim_paras["delta"] >= 0

    # Checks for common returns
    assert optim_paras["coeffs_common"].size == 2

    # Checks for occupations
    assert optim_paras["coeffs_a"].size == 15
    assert optim_paras["coeffs_b"].size == 15
    assert optim_paras["coeffs_edu"].size == 7
    assert optim_paras["coeffs_home"].size == 3

    # Checks shock matrix
    assert optim_paras["shocks_cholesky"].shape == (4, 4)
    np.allclose(optim_paras["shocks_cholesky"], np.tril(optim_paras["shocks_cholesky"]))

    # Checks for type shares
    assert optim_paras["type_shares"].size == num_types * 2

    # Checks for type shifts
    assert optim_paras["type_shifts"].shape == (num_types, 4)

    return True


def _check_parameter_vector(x, a=None):
    """Check optimization parameters."""
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    assert np.all(np.isfinite(x))

    # Check bounds.
    if a is not None:
        for i in range(a["num_paras"]):
            lower, upper = a["optim_paras"]["paras_bounds"][i]
            if lower is not None:
                assert isinstance(lower, float)
                assert lower <= x[i]
                assert abs(lower) < PRINT_FLOAT
            if upper is not None:
                assert isinstance(upper, float)
                assert upper >= x[i]
                assert abs(upper) < PRINT_FLOAT
            if (upper is not None) and (lower is not None):
                assert upper >= lower

    return True
