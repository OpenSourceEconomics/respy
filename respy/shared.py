import json

import numpy as np
import pandas as pd

from respy.config import EXAMPLE_MODELS
from respy.config import HUGE_FLOAT
from respy.config import TEST_RESOURCES_DIR


def get_conditional_probabilities(type_shares, initial_level_of_education):
    """Calculate the conditional choice probabilities.

    The calculation is based on the multinomial logit model for one particular initial
    condition.

    Parameters
    ----------
    type_shares : np.ndarray
        Undocumented parameter.
    initial_level_of_education : np.ndarray
        Array with shape (num_obs,) containing initial levels of education.

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


def cholesky_to_coeffs(shocks_cholesky):
    """Map the Cholesky factor into the coefficients from the .ini file."""
    # TODO: Deprecated.
    shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)
    shocks_cov[np.diag_indices(shocks_cov.shape[0])] **= 0.5
    shocks_coeffs = shocks_cov[np.triu_indices(shocks_cov.shape[0])].tolist()

    return shocks_coeffs


def create_multivariate_standard_normal_draws(num_periods, num_draws, seed):
    """Create the relevant set of draws.

    Handle special case of zero variances as this case is useful for testing.
    The draws are from a standard normal distribution and transformed later in
    the code.

    Parameters
    ----------
    num_periods : int
    num_draws : int
    seed : int

    Returns
    -------
    draws : np.array
        Draws with shape (num_periods, num_draws)

    """
    # Control randomness by setting seed value
    np.random.seed(seed)

    # Draw random deviates from a standard normal distribution.
    draws = np.random.multivariate_normal(
        np.zeros(4), np.identity(4), (num_periods, num_draws)
    )

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

    options_spec = json.loads((TEST_RESOURCES_DIR / f"{model}.json").read_text())
    params_spec = pd.read_csv(TEST_RESOURCES_DIR / f"{model}.csv")

    return params_spec, options_spec
