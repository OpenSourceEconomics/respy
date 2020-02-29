"""Everything related to validate the model."""
import numpy as np


def validate_options(o):
    """Validate the options provided by the user."""
    assert _is_positive_nonzero_integer(o["n_periods"])

    for option, value in o.items():
        if "draws" in option:
            assert _is_positive_nonzero_integer(value)
        elif option.endswith("_seed"):
            assert _is_nonnegative_integer(value)

    assert 0 < o["estimation_tau"]
    assert (
        _is_positive_nonzero_integer(o["interpolation_points"])
        or o["interpolation_points"] == -1
    )
    assert _is_positive_nonzero_integer(o["simulation_agents"])
    assert isinstance(o["core_state_space_filters"], list) and all(
        isinstance(filter_, str) for filter_ in o["core_state_space_filters"]
    )
    assert isinstance(o["inadmissible_states"], dict) and all(
        isinstance(key, str)
        and isinstance(val, list)
        and all(isinstance(condition, str) for condition in val)
        for key, val in o["inadmissible_states"].items()
    )
    assert o["monte_carlo_sequence"] in ["random", "halton", "sobol"]


def validate_params(params, optim_paras):
    """Validate params."""
    _validate_shocks(params, optim_paras)


def _validate_shocks(params, optim_paras):
    """Validate that the elements of the shock matrix are correctly sorted."""
    choices = list(optim_paras["choices"])

    if "shocks_sdcorr" in params.index:
        sds_flat = [f"sd_{c}" for c in choices]

        corrs_flat = []
        for i, c_1 in enumerate(choices):
            for c_2 in choices[: i + 1]:
                if c_1 == c_2:
                    pass
                else:
                    corrs_flat.append(f"corr_{c_1}_{c_2}")
        index = sds_flat + corrs_flat

    else:
        index = []
        for i, c_1 in enumerate(choices):
            for c_2 in choices[: i + 1]:
                if c_1 == c_2:
                    label = "var" if "shocks_cov" in index else "chol"
                    index.append(f"{label}_{c_1}")
                else:
                    label = "cov" if "shocks_cov" in index else "chol"
                    index.append(f"{label}_{c_1}_{c_2}")

    assert all(
        params.filter(regex=r"shocks_(sdcorr|cov|chol)", axis=0).index.get_level_values(
            "name"
        )
        == index
    ), f"Reorder the 'name' index of the shock matrix to {index}."


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
        assert np.all(
            state_space.core.filter(regex=r"\blagged_choice_[0-9]*\b").isin(
                range(len(choices))
            )
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
        _apply_to_attribute_of_state_space(
            state_space.get_attribute("wages"), np.isfinite
        )
    )
    assert np.all(
        _apply_to_attribute_of_state_space(
            state_space.get_attribute("nonpecs"), np.isfinite
        )
    )
    assert np.all(
        _apply_to_attribute_of_state_space(
            state_space.get_attribute("expected_value_functions"), np.isfinite
        )
    )


def _apply_to_attribute_of_state_space(attribute, func):
    """Apply a function to a state space attribute which might be dense or not.

    Attribute might be `state_space.wages` which can be a dictionary or a Numpy array.

    """
    if isinstance(attribute, dict):
        out = [func(val) for val in attribute.values()]
    else:
        out = func(attribute)

    return out
