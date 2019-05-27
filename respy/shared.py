import numpy as np

from respy.config import HUGE_FLOAT


def _paras_parsing_information(num_paras):
    """Dictionary with the start and stop indices of each quantity."""
    num_types = int((num_paras - 53) / 6) + 1
    num_shares = (num_types - 1) * 2
    pinfo = {
        "delta": {"start": 0, "stop": 1},
        "coeffs_common": {"start": 1, "stop": 3},
        "coeffs_a": {"start": 3, "stop": 18},
        "coeffs_b": {"start": 18, "stop": 33},
        "coeffs_edu": {"start": 33, "stop": 40},
        "coeffs_home": {"start": 40, "stop": 43},
        "shocks_coeffs": {"start": 43, "stop": 53},
        "type_shares": {"start": 53, "stop": 53 + num_shares},
        "type_shifts": {"start": 53 + num_shares, "stop": num_paras},
    }
    return pinfo


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
    """ Map the Cholesky factor into the coefficients from the .ini file."""
    shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)
    shocks_cov[np.diag_indices(shocks_cov.shape[0])] **= 0.5
    shocks_coeffs = shocks_cov[np.triu_indices(shocks_cov.shape[0])].tolist()

    return shocks_coeffs


def create_draws(num_periods, num_draws, seed, is_debug):
    """Create the relevant set of draws.

    Handle special case of zero variances as this case is useful for testing.
    The draws are from a standard normal distribution and transformed later in
    the code.

    Parameters
    ----------
    num_periods : int
    num_draws : int
    seed : int
    is_debug : bool

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


def number_of_triangular_elements_to_dimensio(num):
    """Calculate the dimension of a square matrix from number of triangular elements.

    Parameters
    ----------
    num : int
        The number of upper or lower triangular elements in the matrix.

    Example
    -------
    >>> number_of_triangular_elements_to_dimensio(6)
    3
    >>> number_of_triangular_elements_to_dimensio(10)
    4

    """
    return int(np.sqrt(8 * num + 1) / 2 - 0.5)
