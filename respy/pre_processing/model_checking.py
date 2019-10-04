import itertools

import numpy as np
import pandas as pd


def validate_options(o):
    for option, value in o.items():
        if "draws" in option:
            assert _is_positive_nonzero_integer(value)
        elif option.endswith("_seed"):
            assert _is_nonnegative_integer(value)
        elif option.endswith("_seed_startup") or option.endswith("_seed_iteration"):
            assert isinstance(value, itertools.count)
        else:
            pass

    assert 0 < o["estimation_tau"]

    assert (
        _is_positive_nonzero_integer(o["interpolation_points"])
        or o["interpolation_points"] == -1
    )

    # Number of periods.
    assert _is_positive_nonzero_integer(o["n_periods"])

    # Covariates.
    assert all(
        isinstance(key, str) and isinstance(val, str)
        for key, val in o["covariates"].items()
    )


def _is_positive_nonzero_integer(x):
    return isinstance(x, (int, np.integer)) and x > 0


def _is_nonnegative_integer(x):
    return isinstance(x, (int, np.integer)) and x >= 0


def check_model_solution(optim_paras, options, state_space):
    # Distribute class attributes
    choices = optim_paras["choices"]
    max_initial_experience = np.array(
        [choices[choice]["start"].max() for choice in optim_paras["choices_w_exp"]]
    )
    n_initial_exp_comb = np.prod(
        [choices[choice]["start"].shape[0] for choice in optim_paras["choices_w_exp"]]
    )
    n_periods = options["n_periods"]
    n_types = optim_paras["n_types"]
    n_choices_w_exp = len(optim_paras["choices_w_exp"])

    # Check period.
    assert np.all(np.isin(state_space.states[:, 0], range(n_periods)))

    # The sum of years of experiences cannot be larger than constraint time.
    assert np.all(
        state_space.states[:, 1 : n_choices_w_exp + 1].sum(axis=1)
        <= (state_space.states[:, 0] + max_initial_experience.sum())
    )

    # Choice experience cannot exceed the time frame.
    for choice in optim_paras["choices_w_exp"]:
        idx = list(choices).index(choice) + 1
        assert np.all(state_space.states[:, idx] <= choices[choice]["max"])

    # Lagged choices are always in ``range(n_choices)``.
    if optim_paras["n_lagged_choices"]:
        assert np.isin(
            state_space.states[
                :,
                n_choices_w_exp
                + 1 : n_choices_w_exp
                + optim_paras["n_lagged_choices"]
                + 1,
            ],
            range(len(choices)),
        ).all()

    # States and covariates have finite and nonnegative values.
    assert np.all(state_space.states >= 0)
    assert np.all(np.isfinite(state_space.states))

    # Check for duplicate rows in each period. We only have possible duplicates if there
    # are multiple initial conditions.
    assert not pd.DataFrame(state_space.states).duplicated().any()

    # Check the number of states in the first time period.
    obs_factor = np.prod(np.array([len(x) for x in optim_paras["observables"].values()]))
    n_states_start = (
        n_types * n_initial_exp_comb * (optim_paras["n_lagged_choices"] + 1)*obs_factor
    )
    assert state_space.get_attribute_from_period("states", 0).shape[0] == n_states_start
    assert np.sum(state_space.indexer[0] >= 0) == n_states_start

    # Check that we have as many indices as states.
    n_valid_indices = sum((indexer >= 0).sum() for indexer in state_space.indexer)
    assert state_space.states.shape[0] == n_valid_indices

    # Check finiteness of rewards and emaxs.
    assert np.all(np.isfinite(state_space.wages))
    assert np.all(np.isfinite(state_space.nonpec))
    assert np.all(np.isfinite(state_space.emax_value_functions))
