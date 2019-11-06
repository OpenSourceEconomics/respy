"""Contains functions which are shared across other modules.

This module should only import from other packages or modules of respy which also do not
import from respy itself. This is to prevent circular imports.

"""
import numpy as np
import pandas as pd
from numba import njit
from numba import vectorize

from respy.config import HUGE_FLOAT
from respy.config import INADMISSIBILITY_PENALTY


@njit
def aggregate_keane_wolpin_utility(
    wage, nonpec, continuation_value, draw, delta, is_inadmissible
):
    flow_utility = wage * draw + nonpec
    value_function = flow_utility + delta * continuation_value

    if is_inadmissible:
        value_function += INADMISSIBILITY_PENALTY

    return value_function, flow_utility


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


def generate_column_labels_estimation(optim_paras):
    labels = (
        ["Identifier", "Period", "Choice", "Wage"]
        + [f"Experience_{choice.title()}" for choice in optim_paras["choices_w_exp"]]
        + [f"Lagged_Choice_{i}" for i in range(1, optim_paras["n_lagged_choices"] + 1)]
        + [observable.title() for observable in optim_paras["observables"].keys()]
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


def generate_column_labels_simulation(optim_paras):
    est_lab, est_dtypes = generate_column_labels_estimation(optim_paras)
    labels = (
        est_lab
        + ["Type"]
        + [f"Nonpecuniary_Reward_{choice.title()}" for choice in optim_paras["choices"]]
        + [f"Wage_{choice.title()}" for choice in optim_paras["choices_w_wage"]]
        + [f"Flow_Utility_{choice.title()}" for choice in optim_paras["choices"]]
        + [f"Value_Function_{choice.title()}" for choice in optim_paras["choices"]]
        + [f"Shock_Reward_{choice.title()}" for choice in optim_paras["choices"]]
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


def downcast_to_smallest_dtype(series, save_memory=True):
    """Downcast series to smallest possible dtype.

    This function always leaves categoricals unchanged and converts bools to the
    smallest integer dtype. Otherwise, some NumPy arrays will have dtype `object` which
    prevents numerical calculations.

    If `save_memory = True` other dtypes are tested for lower dtypes.

    """
    # We can skip integer as "unsigned" and "signed" will find the same dtypes.
    _downcast_options = ["unsigned", "signed", "float"]

    if series.dtype.name == "category":
        out = series

    elif series.dtype == np.bool:
        out = series.astype(np.dtype("uint8"))

    elif save_memory:
        min_dtype = np.dtype("float64")

        for dc_opt in _downcast_options:
            dtype = pd.to_numeric(series, downcast=dc_opt).dtype

            if dtype.itemsize == 1 and dtype.name.startswith("u"):
                min_dtype = dtype
                break
            elif dtype.itemsize == min_dtype.itemsize and dtype.name.startswith("u"):
                min_dtype = dtype
            elif dtype.itemsize < min_dtype.itemsize:
                min_dtype = dtype
            else:
                pass

        out = series.astype(min_dtype)

    else:
        out = series

    return out


def create_type_covariates(df, optim_paras, options):
    """Create covariates to predict type probabilities.

    In the simulation, the covariates are needed to predict type probabilities and
    assign types to simulated individuals. In the estimation, covariates are necessary
    to weight the probability of observations by type probabilities.

    """
    covariates = create_base_covariates(df, options["covariates"])

    all_data = pd.concat([covariates, df], axis="columns", sort=False)

    all_data = all_data[optim_paras["type_covariates"]].apply(
        downcast_to_smallest_dtype
    )

    return all_data.to_numpy()


def create_base_covariates(states, covariates_spec, raise_errors=True):
    """Create set of covariates for each state.

    Parameters
    ----------
    states : pandas.DataFrame
        DataFrame with some, not all state space dimensions like period, experiences.
    covariates_spec : dict
        Keys represent covariates and values are strings passed to ``df.eval``.
    raise_errors : bool
        Whether to raise errors if a variable was not found. This option is necessary
        for, e.g., :func:`~respy.simulate._get_random_lagged_choices` where not all
        necessary variables exist and it is not clear how to exclude them easily.

    Returns
    -------
    covariates : pandas.DataFrame
        DataFrame with shape (n_states, n_covariates).

    Raises
    ------
    pd.core.computation.ops.UndefinedVariableError
        If a necessary variable is not found in the data.

    """
    covariates = states.copy()

    for covariate, definition in covariates_spec.items():
        try:
            covariates[covariate] = covariates.eval(definition)
        except pd.core.computation.ops.UndefinedVariableError as e:
            if raise_errors:
                raise e
            else:
                pass

    covariates = covariates.drop(columns=states.columns)

    return covariates
