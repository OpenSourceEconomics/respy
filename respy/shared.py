import numpy as np
import pandas as pd
import yaml
from numba import njit

from respy.config import EXAMPLE_MODELS
from respy.config import HUGE_FLOAT
from respy.config import INADMISSIBILITY_PENALTY
from respy.config import TEST_RESOURCES_DIR


@njit
def _aggregate_keane_wolpin_utility(
    wage, nonpec, continuation_value, draw, delta, is_inadmissible
):
    flow_utility = wage * draw + nonpec
    value_function = flow_utility + delta * continuation_value

    if is_inadmissible:
        value_function += INADMISSIBILITY_PENALTY

    return value_function, flow_utility


def get_conditional_probabilities(type_shares, initial_level_of_education):
    """Calculate the conditional choice probabilities.

    The calculation is based on the multinomial logit model for one particular initial
    condition.

    Parameters
    ----------
    type_shares : np.ndarray
        Undocumented parameter.
    initial_level_of_education : np.ndarray
        Array with shape (n_obs,) containing initial levels of education.

    """
    type_shares = type_shares.reshape(-1, 2)
    covariates = np.column_stack(
        (np.ones(initial_level_of_education.shape[0]), initial_level_of_education > 9)
    )
    probs = np.exp(covariates.dot(type_shares.T))
    probs /= probs.sum(axis=1, keepdims=True)

    if initial_level_of_education.shape[0] == 1:
        probs = probs.ravel()

    return probs


def create_base_draws(shape, seed):
    """Create the relevant set of draws.

    Handle special case of zero variances as this case is useful for testing.
    The draws are from a standard normal distribution and transformed later in
    the code.

    Parameters
    ----------
    shape : Tuple[int]
        Tuple representing the shape of the resulting array.
    seed : int
        Seed to control randomness.

    Returns
    -------
    draws : np.array
        Draws with shape (n_periods, n_draws)

    """
    # Control randomness by setting seed value
    np.random.seed(seed)

    # Draw random deviates from a standard normal distribution.
    draws = np.random.standard_normal(shape)

    return draws


def transform_disturbances(draws, shocks_mean, shocks_cholesky):
    """Transform the standard normal deviates to the relevant distribution."""
    draws_transformed = draws.dot(shocks_cholesky.T)

    draws_transformed += shocks_mean

    draws_transformed[:, :2] = np.clip(
        np.exp(draws_transformed[:, :2]), 0.0, HUGE_FLOAT
    )

    return draws_transformed


def get_example_model(model):
    assert model in EXAMPLE_MODELS, f"{model} is not in {EXAMPLE_MODELS}."

    options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
    params = pd.read_csv(TEST_RESOURCES_DIR / f"{model}.csv")

    return params, options


def _generate_column_labels_estimation(options):
    labels = (
        ["Identifier", "Period", "Choice", "Wage"]
        + [f"Experience_{sec.title()}" for sec in options["choices_w_exp"]]
        + ["Lagged_Choice"]
    )

    dtypes = {}
    for label in labels:
        if label == "Wage":
            dtypes[label] = float
        elif "Choice" in label:
            dtypes[label] = "category"
        else:
            dtypes[label] = int

    return labels, dtypes


def _generate_column_labels_simulation(options):
    est_lab, est_dtypes = _generate_column_labels_estimation(options)
    labels = (
        est_lab
        + ["Type"]
        + [f"Nonpecuniary_Reward_{sec.title()}" for sec in options["choices"]]
        + [f"Wages_{sec.title()}" for sec in options["choices_w_wage"]]
        + [f"Flow_Utility_{sec.title()}" for sec in options["choices"]]
        + [f"Value_Function_{sec.title()}" for sec in options["choices"]]
        + [f"Shock_Reward_{sec.title()}" for sec in options["choices"]]
        + ["Discount_Rate"]
    )

    dtypes = {col: (int if col == "Type" else float) for col in labels}
    dtypes = {**dtypes, **est_dtypes}

    return labels, dtypes
