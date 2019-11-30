"""Contains functions which are shared across other modules.

This module should only import from other packages or modules of respy which also do not
import from respy itself. This is to prevent circular imports.

"""
import chaospy as cp
import numba as nb
import numpy as np
import pandas as pd

from respy.config import HUGE_FLOAT
from respy.config import INADMISSIBILITY_PENALTY


@nb.njit
def aggregate_keane_wolpin_utility(
    wage, nonpec, continuation_value, draw, delta, is_inadmissible
):
    flow_utility = wage * draw + nonpec
    value_function = flow_utility + delta * continuation_value

    if is_inadmissible:
        value_function += INADMISSIBILITY_PENALTY

    return value_function, flow_utility


def create_base_draws(shape, seed, monte_carlo_sequence):
    """Create a set of draws from the standard normal distribution.

    The draws are either drawn randomly or from quasi-random low-discrepancy sequences,
    i.e., Sobol or Halton.

    `"random"` is used to draw random standard normal shocks for the Monte Carlo
    integrations or because individuals face random shocks in the simulation.

    `"halton"` or `"sobol"` can be used to change the sequence for two Monte Carlo
    integrations. First, the calculation of the expected value function (EMAX) in the
    solution and the choice probabilities in the maximum likelihood estimation.

    For the solution and estimation it is necessary to have the same randomness in every
    iteration. Otherwise, there is chatter in the simulation, i.e. a difference in
    simulated values not only due to different parameters but also due to draws (see
    10.5 in [1]_). At the same time, the variance-covariance matrix of the shocks is
    estimated along all other parameters and changes every iteration. Thus, instead of
    sampling draws from a varying multivariate normal distribution, standard normal
    draws are sampled here and transformed to the distribution specified by the
    parameters in
    :func:`transform_base_draws_with_cholesky_factor`.

    Parameters
    ----------
    shape : tuple(int)
        Tuple representing the shape of the resulting array.
    seed : int
        Seed to control randomness.
    monte_carlo_sequence : {"random", "halton", "sobol"}
        Name of the sequence.

    Returns
    -------
    draws : numpy.ndarray
        Array with shape (n_choices, n_draws, n_choices).

    See also
    --------
    transform_base_draws_with_cholesky_factor

    References
    ----------
    .. [1] Train, K. (2009). `Discrete Choice Methods with Simulation
           <https://eml.berkeley.edu/books/choice2.html>`_. *Cambridge: Cambridge
           University Press.*

    """
    n_choices = shape[2]
    n_points = shape[0] * shape[1]

    np.random.seed(seed)

    if monte_carlo_sequence == "random":
        draws = np.random.standard_normal(shape)

    elif monte_carlo_sequence == "halton":
        distribution = cp.MvNormal(loc=np.zeros(n_choices), scale=np.eye(n_choices))
        draws = distribution.sample(n_points, rule="H").T.reshape(shape)

    elif monte_carlo_sequence == "sobol":
        distribution = cp.MvNormal(loc=np.zeros(n_choices), scale=np.eye(n_choices))
        draws = distribution.sample(n_points, rule="S").T.reshape(shape)

    else:
        raise NotImplementedError

    return draws


def transform_base_draws_with_cholesky_factor(
    draws, shocks_mean, shocks_cholesky, n_wages
):
    r"""Transform standard normal draws with the Cholesky factor.

    The standard normal draws are transformed to normal draws with variance-covariance
    matrix :math:`\Sigma` by multiplication with the Cholesky factor :math:`L` where
    :math:`L^TL = \Sigma`. See chapter 7.4 in [1]_ for more information.

    This function relates to :func:`create_base_draws` in the sense that it transforms
    the unchanging standard normal draws to the distribution with the
    variance-covariance matrix specified by the parameters.

    References
    ----------
    .. [1] Gentle, J. E. (2009). Computational statistics (Vol. 308). New York:
           Springer.

    See also
    --------
    create_base_draws

    """
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
        + [observable.title() for observable in optim_paras["observables"]]
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


@nb.njit
def clip(x, minimum=None, maximum=None):
    """Clip input array at minimum and maximum."""
    out = np.empty_like(x)

    for index, value in np.ndenumerate(x):
        if minimum is not None and value < minimum:
            out[index] = minimum
        elif maximum is not None and value > maximum:
            out[index] = maximum
        else:
            out[index] = value

    return out


def downcast_to_smallest_dtype(series):
    # We can skip integer as "unsigned" and "signed" will find the same dtypes.
    _downcast_options = ["unsigned", "signed", "float"]

    if series.dtype.name == "category":
        min_dtype = "category"

    elif series.dtype == np.bool:
        min_dtype = np.dtype("uint8")

    else:
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

    return series.astype(min_dtype)


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


def convert_choice_variables_from_categorical_to_codes(df, optim_paras):
    """Recode choices to choice codes in the model.

    We cannot use ``.cat.codes`` because order might be different. The model requires an
    order of ``choices_w_exp_w_wag``, ``choices_w_exp_wo_wage``,
    ``choices_wo_exp_wo_wage``.

    See also
    --------
    respy.pre_processing.model_processing._order_choices

    """
    choices_to_codes = {choice: i for i, choice in enumerate(optim_paras["choices"])}

    if "choice" in df.columns:
        df.choice = df.choice.replace(choices_to_codes).astype(np.uint8)

    for i in range(1, optim_paras["n_lagged_choices"] + 1):
        label = f"lagged_choice_{i}"
        if label in df.columns:
            df[label] = df[label].replace(choices_to_codes).astype(np.uint8)

    return df


def normalize_probabilities(probabilities):
    """Normalize probabilities such that their sum equals one.

    Example
    -------
    The following `probs` do not sum to one after dividing by the sum.

    >>> probs = np.array([0.3775843411510946, 0.5384246942799851, 0.6522988820635421])
    >>> normalize_probabilities(probs)
    array([0.24075906, 0.34331568, 0.41592526])

    """
    probabilities = probabilities / np.sum(probabilities)
    probabilities[-1] = 1 - probabilities[:-1].sum()

    return probabilities


@nb.guvectorize(
    ["f8, f8, f8, f8, f8, b1, f8[:], f8[:]"],
    "(), (), (), (), (), () -> (), ()",
    nopython=True,
    target="parallel",
)
def calculate_value_functions_and_flow_utilities(
    wage,
    nonpec,
    continuation_value,
    draw,
    delta,
    is_inadmissible,
    value_function,
    flow_utility,
):
    """Calculate the choice-specific value functions and flow utilities.

    To apply :func:`aggregate_keane_wolpin_utility` to arrays with arbitrary dimensions,
    this function uses :func:`numba.guvectorize`. One cannot use :func:`numba.vectorize`
    because it does not support multiple return values.

    See also
    --------
    aggregate_keane_wolpin_utility

    """
    value_function[0], flow_utility[0] = aggregate_keane_wolpin_utility(
        wage, nonpec, continuation_value, draw, delta, is_inadmissible
    )
