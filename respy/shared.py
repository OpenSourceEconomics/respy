"""Contains functions which are shared across other modules.

This module should only import from other packages or modules of respy which also do not
import from respy itself. This is to prevent circular imports.

"""
import chaospy as cp
import numba as nb
import numpy as np
import pandas as pd

from respy.config import INADMISSIBILITY_PENALTY
from respy.config import MAX_LOG_FLOAT
from respy.config import MIN_LOG_FLOAT


@nb.njit
def aggregate_keane_wolpin_utility(
    wage, nonpec, continuation_value, draw, delta, is_inadmissible
):
    """Calculate the utility of Keane and Wolpin models.

    Note that the function works for working and non-working alternatives as wages are
    set to one for non-working alternatives such that the draws enter the utility
    function additively.

    Parameters
    ----------
    wage : float
        Value of the wage component. Note that for non-working alternatives this value
        is actually zero, but to simplify computations it is set to one.
    nonpec : float
        Value of the non-pecuniary component.
    continuation_value : float
        Value of the continuation value which is the expected present-value of the
        following state.
    draw : float
        The shock which enters the enters the reward of working alternatives
        multiplicatively and of non-working alternatives additively.
    delta : float
        The discount factor to calculate the present value of continuation values.
    is_inadmissible : float
        An indicator for whether the choice is in the current choice set.

    Returns
    -------
    alternative_specific_value_function : float
        The expected present value of an alternative.
    flow_utility : float
        The immediate reward of an alternative.

    """
    flow_utility = wage * draw + nonpec
    alternative_specific_value_function = flow_utility + delta * continuation_value

    if is_inadmissible:
        alternative_specific_value_function += INADMISSIBILITY_PENALTY

    return alternative_specific_value_function, flow_utility


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
    parameters in :func:`transform_base_draws_with_cholesky_factor`.

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
    n_choices = shape[-1]
    n_points = np.prod(shape[:-1])

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


def transform_base_draws_with_cholesky_factor(draws, shocks_cholesky, n_wages):
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
    draws_transformed[..., :n_wages] = np.exp(
        np.clip(draws_transformed[..., :n_wages], MIN_LOG_FLOAT, MAX_LOG_FLOAT)
    )

    return draws_transformed


def generate_column_dtype_dict_for_estimation(optim_paras):
    """Generate column labels for data necessary for the estimation."""
    labels = (
        ["Identifier", "Period", "Choice", "Wage"]
        + [f"Experience_{choice.title()}" for choice in optim_paras["choices_w_exp"]]
        + [f"Lagged_Choice_{i}" for i in range(1, optim_paras["n_lagged_choices"] + 1)]
        + [observable.title() for observable in optim_paras["observables"]]
    )

    column_dtype_dict = {}
    for label in labels:
        if label == "Wage":
            column_dtype_dict[label] = float
        elif "Choice" in label:
            column_dtype_dict[label] = "category"
        else:
            column_dtype_dict[label] = int

    return column_dtype_dict


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


def downcast_to_smallest_dtype(series, downcast_options=None):
    """Downcast the dtype of a :class:`pandas.Series` to the lowest possible dtype.

    By default, variables are converted to signed or unsigned integers. Use ``"float"``
    to cast variables from ``float64`` to ``float32``.

    Be aware that NumPy integers silently overflow which is why conversion to low dtypes
    should be done after calculations. For example, using :class:`numpy.uint8` for an
    array and squaring the elements leads to silent overflows for numbers higher than
    255.

    For more information on the dtype boundaries see the NumPy documentation under
    https://docs.scipy.org/doc/numpy-1.17.0/user/basics.types.html.

    """
    # We can skip integer as "unsigned" and "signed" will find the same dtypes.
    if downcast_options is None:
        downcast_options = ["unsigned", "signed"]

    if series.dtype.name == "category":
        out = series

    # Convert bools to integers because they turn the dot product in
    # `create_choice_rewards` to the object dtype.
    elif series.dtype == np.bool:
        out = series.astype(np.dtype("uint8"))

    else:
        min_dtype = series.dtype

        for dc_opt in downcast_options:
            try:
                dtype = pd.to_numeric(series, downcast=dc_opt).dtype
            # A ValueError happens if strings are found in the series.
            except ValueError:
                min_dtype = "category"
                break

            # If we can convert the series to an unsigned integer, we can stop.
            if dtype.name.startswith("u"):
                min_dtype = dtype
                break
            elif dtype.itemsize < min_dtype.itemsize:
                min_dtype = dtype
            else:
                pass

        out = series.astype(min_dtype)

    return out


def cast_bool_to_numeric(df):
    """Cast columns with boolean data type to the smallest integer."""
    bool_columns = df.columns[df.dtypes == np.bool]
    for column in bool_columns:
        df[column] = df[column].astype(np.uint8)
    return df


def compute_covariates(df, definitions, raise_errors=True):
    """Compute covariates.

    The function iterates over the definitions of covariates and tries to compute them.
    It keeps track on how many covariates still need to be computed and stops if the
    number does not change anymore. This might be due to missing information.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with some, maybe not all state space dimensions like period,
        experiences.
    definitions : dict
        Keys represent covariates and values are strings passed to ``df.eval``.
    raise_errors : bool, default True
        Whether to raise errors if variables cannot be computed. This option is
        necessary for, e.g., :func:`~respy.simulate._sample_characteristic` where not
        all necessary variables exist and it is not easy to exclude covariates which
        depend on them.

    Returns
    -------
    covariates : pandas.DataFrame
        DataFrame with shape (n_states, n_covariates).

    Raises
    ------
    Exception
        If variables cannot be computed and ``raise_errors`` is true.

    """
    has_covariates_left_changed = True
    covariates_left = list(definitions)

    while has_covariates_left_changed:
        n_covariates_left = len(covariates_left)

        # Create a copy of `covariates_left` to remove elements without side-effects.
        for covariate in covariates_left.copy():
            # Check if the covariate does not exist and needs to be computed.
            is_covariate_missing = covariate not in df.columns
            # Check that the dependencies are present and do not contain NaNs.
            index_or_columns = df.columns.union(df.index.names)
            are_dependencies_present = all(
                df.eval(f"{dep}.notna().all()") if dep in index_or_columns else False
                for dep in definitions[covariate]["depends_on"]
            )

            if not is_covariate_missing:
                covariates_left.remove(covariate)
            elif is_covariate_missing and are_dependencies_present:
                df[covariate] = df.eval(definitions[covariate]["formula"])
                covariates_left.remove(covariate)

        has_covariates_left_changed = n_covariates_left != len(covariates_left)

    if covariates_left and raise_errors:
        raise Exception(f"Cannot compute all covariates: {covariates_left}.")

    return df


def convert_labeled_variables_to_codes(df, optim_paras):
    """Convert labeled variables to codes.

    We need to check choice variables and observables for potential labels. The
    mapping from labels to code can be inferred from the order in ``optim_paras``.

    """
    choices_to_codes = {choice: i for i, choice in enumerate(optim_paras["choices"])}

    if "choice" in df.columns:
        df.choice = df.choice.replace(choices_to_codes).astype(np.uint8)

    for i in range(1, optim_paras["n_lagged_choices"] + 1):
        label = f"lagged_choice_{i}"
        if label in df.columns:
            df[label] = df[label].replace(choices_to_codes).astype(np.uint8)

    observables = optim_paras["observables"]
    for observable in observables:
        if observable in df.columns:
            levels_to_codes = {lev: i for i, lev in enumerate(observables[observable])}
            df[observable] = df[observable].replace(levels_to_codes).astype(np.uint8)

    return df


def rename_labels_to_internal(x):
    """Shorten labels and convert them to lower-case."""
    return x.replace("Experience", "exp").lower()


def rename_labels_from_internal(x):
    """Shorten labels and convert them to lower-case."""
    return x.replace("exp", "Experience").title()


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


def create_core_state_space_columns(optim_paras):
    """Create internal column names for the core state space."""
    return [f"exp_{choice}" for choice in optim_paras["choices_w_exp"]] + [
        f"lagged_choice_{i}" for i in range(1, optim_paras["n_lagged_choices"] + 1)
    ]


def create_dense_state_space_columns(optim_paras):
    """Create internal column names for the dense state space."""
    columns = list(optim_paras["observables"])
    if optim_paras["n_types"] >= 2:
        columns += ["type"]

    return columns


def create_state_space_columns(optim_paras):
    """Create names of state space dimensions excluding the period and identifier."""
    return create_core_state_space_columns(
        optim_paras
    ) + create_dense_state_space_columns(optim_paras)


def apply_to_state_space_attribute(attribute, func):
    """Apply a function to a state space attribute which might be dense or not."""
    if isinstance(attribute, dict):
        out = [func(val) for val in attribute.values()]
    else:
        out = func(attribute)

    return out
