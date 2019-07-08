import numpy as np
import pandas as pd
import yaml
from numba import guvectorize
from numba import njit
from numba import vectorize

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


@guvectorize(
    ["f8[:, :], f8[:], f8[:]"],
    "(n_types, n_covariates), (n_covariates) -> (n_types)",
    nopython=True,
    target="parallel",
)
def predict_multinomial_logit(type_proportions, type_covariates, probs):
    """Predict probabilities based on a multinomial logit regression.

    The function is used to predict the probability for all types based on the initial
    characteristics of an individual. This is necessary to sample initial types in the
    simulation and to weight the probability of an observation by types in the
    estimation.

    The `multinomial logit model
    <https://en.wikipedia.org/wiki/Multinomial_logistic_regression>`_  predicts the
    probability that an individual belongs to a certain type. The sum over all
    type-probabilities is one.

    Parameters
    ----------
    type_proportions : numpy.ndarray
        Array with shape (n_types, n_type_covariates) containing parameters to predict
        type probabilities.
    type_covariates : numpy.ndarray
        Array with shape (n_type_covariates,) containing type covariates.

    Returns
    -------
    probs : numpy.ndarray
        Array with shape (n_types,) containing the probabilities for each type.

    """
    n_types, n_covariates = type_proportions.shape

    denominator = 0

    for type_ in range(n_types):
        prob_type = 0
        for cov in range(n_covariates):
            prob_type += type_proportions[type_, cov] * type_covariates[cov]

        exp_prob_type = np.exp(prob_type)
        probs[type_] = exp_prob_type
        denominator += exp_prob_type

    probs /= denominator


def create_base_draws(shape, seed):
    """Create the relevant set of draws.

    Handle special case of zero variances as this case is useful for testing.
    The draws are from a standard normal distribution and transformed later in
    the code.

    Parameters
    ----------
    shape : tuple(int)
        Tuple representing the shape of the resulting array.
    seed : int
        Seed to control randomness.

    Returns
    -------
    draws : numpy.ndarray
        Draws with shape (n_periods, n_draws)

    """
    # Control randomness by setting seed value
    np.random.seed(seed)

    # Draw random deviates from a standard normal distribution.
    draws = np.random.standard_normal(shape)

    return draws


def transform_disturbances(draws, shocks_mean, shocks_cholesky, n_wages):
    """Transform the standard normal deviates to the relevant distribution."""
    draws_transformed = draws.dot(shocks_cholesky.T)

    draws_transformed += shocks_mean

    draws_transformed[:, :n_wages] = np.clip(
        np.exp(draws_transformed[:, :n_wages]), 0.0, HUGE_FLOAT
    )

    return draws_transformed


def get_example_model(model):
    assert model in EXAMPLE_MODELS, f"{model} is not in {EXAMPLE_MODELS}."

    options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
    params = pd.read_csv(
        TEST_RESOURCES_DIR / f"{model}.csv", index_col=["category", "name"]
    )

    return params, options


def _generate_column_labels_estimation(options):
    labels = (
        ["Identifier", "Period", "Choice", "Wage"]
        + [f"Experience_{choice.title()}" for choice in options["choices_w_exp"]]
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
        + [f"Nonpecuniary_Reward_{choice.title()}" for choice in options["choices"]]
        + [f"Wage_{choice.title()}" for choice in options["choices_w_wage"]]
        + [f"Flow_Utility_{choice.title()}" for choice in options["choices"]]
        + [f"Value_Function_{choice.title()}" for choice in options["choices"]]
        + [f"Shock_Reward_{choice.title()}" for choice in options["choices"]]
        + ["Discount_Rate"]
    )

    dtypes = {col: (int if col == "Type" else float) for col in labels}
    dtypes = {**dtypes, **est_dtypes}

    return labels, dtypes


@vectorize("f8(f8, f8, f8)", nopython=True, target="cpu")
def clip(x, minimum=None, maximum=None):
    """Clip (limit) input value.

    Parameters
    ----------
    x : float
        Value to be clipped.
    minimum : float
        Lower limit.
    maximum : float
        Upper limit.

    Returns
    -------
    float
        Clipped value.

    """
    if minimum is not None and x < minimum:
        return minimum
    elif maximum is not None and x > maximum:
        return maximum
    else:
        return x
