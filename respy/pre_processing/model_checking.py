"""Everything related to validate the model."""
import itertools

import numpy as np

from respy.shared import apply_to_state_space_attribute


def validate_options(o):
    """Validate the options provided by the user."""
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


def _is_positive_nonzero_integer(x):
    return isinstance(x, (int, np.integer)) and x > 0


def _is_nonnegative_integer(x):
    return isinstance(x, (int, np.integer)) and x >= 0


def check_model_solution(optim_paras, options, state_space):
    """Check properties of the solution of a model."""
    # Distribute class attributes
    choices = optim_paras["choices"]
    max_initial_experience = np.array(
        [max(choices[choice]["start"]) for choice in optim_paras["choices_w_exp"]]
    )
    n_periods = options["n_periods"]

    # Check period.
    assert np.all(np.isin(state_space.core.period, range(n_periods)))

    # The sum of years of experiences cannot be larger than constraint time.
    assert np.all(
        state_space.core[[f"exp_{c}" for c in optim_paras["choices_w_exp"]]].sum(axis=1)
        <= (state_space.core.period + max_initial_experience.sum())
    )

    # Choice experience cannot exceed the time frame.
    for choice in optim_paras["choices_w_exp"]:
        assert state_space.core[f"exp_{choice}"].le(choices[choice]["max"]).all()

    # Lagged choices are always in ``range(n_choices)``.
    if optim_paras["n_lagged_choices"]:
        assert (
            state_space.core.filter(regex=r"\blagged_choice_[0-9]*\b")
            .isin(range(len(choices)))
            .all()
            .all()
        )

    assert np.all(np.isfinite(state_space.core))

    # Check for duplicate rows in each period. We only have possible duplicates if there
    # are multiple initial conditions.
    assert not state_space.core.duplicated().any()

    # Check that we have as many indices as states.
    n_valid_indices = sum((indexer >= 0).sum() for indexer in state_space.indexer)
    assert state_space.core.shape[0] == n_valid_indices

    # Check finiteness of rewards and emaxs.
    assert np.all(
        apply_to_state_space_attribute(state_space.get_attribute("wages"), np.isfinite)
    )
    assert np.all(
        apply_to_state_space_attribute(
            state_space.get_attribute("nonpecs"), np.isfinite
        )
    )
    assert np.all(
        apply_to_state_space_attribute(
            state_space.get_attribute("expected_value_functions"), np.isfinite
        )
    )
