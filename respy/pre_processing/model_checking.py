import numpy as np
import pandas as pd


def _validate_params(params):
    # TODO: Ensure that the shock matrix is correctly ordered.

    types = sorted(
        params.index.get_level_values(0)
        .str.extract(r"(\btype_[0-9]+\b)")[0]
        .dropna()
        .unique()
    )
    if types:
        # Ensure that parameters to predict type probabilities have the same ordering
        # for every type.
        covariates = params.loc[types[0]].get_level_values(0)
        for type_ in types[1:]:
            assert all(covariates == params.loc[type_].get_level_values(0))

        # TODO: Ensure that type shifts are correctly ordered.


def _validate_options(o):
    # Choices with experience.
    choices = o["choices"]
    for choice in o["choices_w_exp"]:
        assert (
            len(choices[choice]["lagged"])
            == len(choices[choice]["share"])
            == len(choices[choice]["start"])
        )
        assert isinstance(choices[choice]["lagged"], np.ndarray)
        assert isinstance(choices[choice]["share"], np.ndarray)
        assert isinstance(choices[choice]["start"], np.ndarray)
        assert all(0 <= item <= 1 for item in choices[choice]["lagged"])
        assert all(0 <= item <= 1 for item in choices[choice]["share"])
        assert np.isclose(choices[choice]["share"].sum(), 1)
        assert all(_is_nonnegative_integer(i) for i in choices[choice]["start"])
        assert _is_nonnegative_integer(choices[choice]["max"])
        assert all(i <= choices[choice]["max"] for i in choices[choice]["start"])

    # Estimation.
    assert _is_positive_nonzero_integer(o["estimation_draws"])

    assert _is_positive_nonzero_integer(o["estimation_seed"])
    assert 0 <= o["estimation_tau"]

    # Interpolation.
    assert (
        _is_positive_nonzero_integer(o["interpolation_points"])
        or o["interpolation_points"] == -1
    )

    # Number of periods.
    assert _is_positive_nonzero_integer(o["n_periods"])

    # Simulation.
    assert _is_positive_nonzero_integer(o["simulation_agents"])
    assert _is_positive_nonzero_integer(o["simulation_seed"])

    # Solution.
    assert _is_positive_nonzero_integer(o["solution_draws"])
    assert _is_positive_nonzero_integer(o["solution_seed"])

    # Covariates.
    assert isinstance(o["covariates"], dict)
    assert all(
        isinstance(key, str) and isinstance(val, str)
        for key, val in o["covariates"].items()
    )

    # Every choice must have a restriction.
    assert all(i in o["inadmissible_states"] for i in o["choices"])


def _is_positive_nonzero_integer(x):
    return isinstance(x, (int, np.integer)) and x > 0


def _is_nonnegative_integer(x):
    return isinstance(x, (int, np.integer)) and x >= 0


def check_model_solution(options, state_space):
    # Distribute class attributes
    edu_start = options["choices"]["edu"]["start"]
    n_initial_exp_edu = len(edu_start)
    edu_start_max = max(edu_start)
    n_periods = options["n_periods"]
    n_types = options["n_types"]

    # Check period.
    assert np.all(np.isin(state_space.states[:, 0], range(n_periods)))

    # The sum of years of experiences cannot be larger than constraint time.
    assert np.all(
        state_space.states[:, 1 : len(options["choices_w_exp"]) + 1].sum(axis=1)
        <= (state_space.states[:, 0] + edu_start_max)
    )

    # Choice experience cannot exceed the time frame.
    for choice in options["choices_w_exp"]:
        idx = list(options["choices"]).index(choice) + 1
        assert np.all(state_space.states[:, idx] <= options["choices"][choice]["max"])

    # Lagged choices are always between one and four.
    assert np.isin(state_space.states[:, -2], range(len(options["choices"]))).all()

    # States and covariates have finite and nonnegative values.
    assert np.all(state_space.states >= 0)
    assert np.all(np.isfinite(state_space.states))

    # Check for duplicate rows in each period. We only have possible duplicates if there
    # are multiple initial conditions.
    assert not pd.DataFrame(state_space.states).duplicated().any()

    # Check the number of states in the first time period.
    n_states_start = n_types * n_initial_exp_edu * 2
    assert state_space.get_attribute_from_period("states", 0).shape[0] == n_states_start
    assert np.sum(state_space.indexer[0] >= 0) == n_states_start

    # Check that we have as many indices as states.
    assert state_space.states.shape[0] == (state_space.indexer >= 0).sum()

    # Check finiteness of rewards and emaxs.
    assert np.all(np.isfinite(state_space.wages))
    assert np.all(np.isfinite(state_space.nonpec))
    assert np.all(np.isfinite(state_space.continuation_values))
    assert np.all(np.isfinite(state_space.emax_value_functions))
