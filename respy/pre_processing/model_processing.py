"""Process model specification files or objects."""
import copy
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from estimagic.optimization.utilities import chol_params_to_lower_triangular_matrix
from estimagic.optimization.utilities import cov_params_to_matrix
from estimagic.optimization.utilities import robust_cholesky
from estimagic.optimization.utilities import sdcorr_params_to_matrix

from respy.config import DEFAULT_OPTIONS
from respy.config import SEED_STARTUP_ITERATION_GAP
from respy.pre_processing.model_checking import validate_options

warnings.simplefilter("error", category=pd.errors.PerformanceWarning)


def process_params_and_options(params, options):
    options = _read_options(options)
    options = {**DEFAULT_OPTIONS, **options}
    options = _create_internal_seeds_from_user_seeds(options)
    validate_options(options)

    params = _read_params(params)
    optim_paras = _parse_parameters(params, options)

    optim_paras["n_periods"] = options["n_periods"]

    return optim_paras, options


def _read_options(input_):
    if not isinstance(input_, (Path, dict)):
        raise TypeError("options must be pathlib.Path or dictionary.")

    if isinstance(input_, Path):
        options = yaml.safe_load(input_.read_text())
    else:
        options = copy.deepcopy(input_)

    return options


def _create_internal_seeds_from_user_seeds(options):
    """Create internal seeds from user input.

    Instead of reusing the same seed, we use sequences of seeds incrementing by one. It
    ensures that we do not accidentally draw the same randomness twice.

    Furthermore, we need to sets of seeds. The first set is for building
    :func:`~respy.simulate.simulate` or :func:`~respy.likelihood.log_like` where
    ``"startup"`` seeds are used to generate the draws. The second set is for the
    iterations and has to be reset to the initial value at the beginning of every
    iteration.

    See :ref:`randomness-and-reproducibility` for more information.

    """

    seeds = [f"{key}_seed" for key in ["solution", "simulation", "estimation"]]

    # Check that two seeds are not equal. Otherwise, raise warning.
    if (np.bincount([options[seed] for seed in seeds]) > 1).any():
        warnings.warn("All seeds should be different.", category=UserWarning)

    for seed, start, end in zip(
        seeds, [100, 10_000, 1_000_000], [1_000, 100_000, 10_000_000]
    ):
        np.random.seed(options[seed])
        seed_startup = np.random.randint(start, end)
        options[f"{seed}_startup"] = itertools.count(seed_startup)
        seed_iteration = seed_startup + SEED_STARTUP_ITERATION_GAP
        options[f"{seed}_iteration"] = itertools.count(seed_iteration)

    return options


def _read_params(input_):
    input_ = pd.read_csv(input_) if isinstance(input_, Path) else input_.copy()

    if isinstance(input_, pd.DataFrame):
        if not input_.index.names == ["category", "name"]:
            input_.set_index(["category", "name"], inplace=True)
        params = input_["value"]
    elif isinstance(input_, pd.Series):
        params = input_
        assert params.index.names == [
            "category",
            "name",
        ], "params as pd.Series has wrong index."
    else:
        raise TypeError("params must be Path, pd.DataFrame or pd.Series.")

    return params


def _parse_parameters(params, options):
    """Parse the parameter vector into a dictionary of model quantities."""
    optim_paras = {}

    optim_paras["delta"] = params.loc[("delta", "delta")]
    optim_paras = _parse_observables(optim_paras, params)
    optim_paras = _parse_choices(optim_paras, params, options)
    optim_paras = _parse_choice_parameters(optim_paras, params)
    optim_paras = _parse_initial_and_max_experience(optim_paras, params, options)
    optim_paras = _parse_shocks(optim_paras, params)
    optim_paras = _parse_measurement_errors(optim_paras, params)
    optim_paras = _parse_types(optim_paras, params)
    optim_paras = _parse_lagged_choices(optim_paras, options, params)

    return optim_paras


def _parse_choices(optim_paras, params, options):
    """Define unique order of choices.

    This function defines a unique order of choices. Choices can be separated in choices
    with experience and wage, with experience but without wage and without experience
    and wage. This distinction is used to create a unique ordering of choices. Within
    each group, we order alphabetically.

    """
    # Be careful with ``choices_w_exp_fuzzy`` as it contains some erroneous elements,
    # e.g., ``"a_squared"`` from the covariate ``"exp_a_squared"``.
    choices_w_exp_fuzzy = set(_infer_choices_with_experience(params, options))
    choices_w_wage = set(_infer_choices_with_prefix(params, "wage"))
    choices_w_nonpec = set(_infer_choices_with_prefix(params, "nonpec"))
    choices_w_exp_wo_wage = (choices_w_exp_fuzzy & choices_w_nonpec) - choices_w_wage
    choices_wo_exp_wo_wage = choices_w_nonpec - choices_w_exp_fuzzy

    optim_paras["choices_w_wage"] = sorted(choices_w_wage)
    optim_paras["choices_w_exp"] = sorted(choices_w_wage) + sorted(
        choices_w_exp_wo_wage
    )
    optim_paras["choices_wo_exp"] = sorted(choices_wo_exp_wo_wage)

    # Dictionaries are insertion ordered since Python 3.6+.
    order = optim_paras["choices_w_exp"] + optim_paras["choices_wo_exp"]
    optim_paras["choices"] = {choice: {} for choice in order}

    return optim_paras


def _parse_observables(optim_paras, params):
    if "observables" in params.index.get_level_values(0):
        obs = [x[1][:-2] for x in params.index if "observables" in x[0]]
        optim_paras["observables"] = {
            x: np.array(
                [params.loc[(f"observables", f"{x}_{y}")] for y in range(obs.count(x))]
            )
            for x in set(obs)
        }
    else:
        optim_paras["observables"] = {}
    return optim_paras


def _parse_choice_parameters(optim_paras, params):
    """Parse utility parameters for choices."""
    for choice in optim_paras["choices"]:
        if f"wage_{choice}" in params.index:
            optim_paras[f"wage_{choice}"] = params.loc[f"wage_{choice}"]
        if f"nonpec_{choice}" in params.index:
            optim_paras[f"nonpec_{choice}"] = params.loc[f"nonpec_{choice}"]

    return optim_paras


def _parse_initial_and_max_experience(optim_paras, params, options):
    """Process initial experience distributions and maximum experience.

    Each choice with experience accumulation has to be defined in three ways. First, a
    set of initial experience values, second, the share of the initial experience levels
    in the population and the maximum of accumulated experience. The defaults are zero
    experience with no upper limit.

    """
    for choice in optim_paras["choices_w_exp"]:
        if f"initial_exp_{choice}" in params.index:
            # Take starts and shares and convert index to numeric or sorting will fail.
            # Maybe waiting on https://github.com/pandas-dev/pandas/pull/27237.
            starts_and_shares = params.loc[f"initial_exp_{choice}"].copy()
            starts_and_shares.index = starts_and_shares.index.astype(np.uint8)
            starts_and_shares.sort_index(inplace=True)
            starts = starts_and_shares.index.to_numpy()
            shares = starts_and_shares.to_numpy()
            if shares.sum() != 1:
                warnings.warn(
                    f"The shares of initial experiences for choice '{choice}' do not "
                    "sum to one. Shares are divided by their sum for normalization.",
                    category=UserWarning,
                )
                shares = shares / shares.sum()
        else:
            starts = np.zeros(1, dtype=np.uint8)
            shares = np.ones(1, dtype=np.uint8)

        max_ = int(params.get(("maximum_exp", choice), options["n_periods"] - 1))

        optim_paras["choices"][choice]["start"] = starts
        optim_paras["choices"][choice]["share"] = shares
        optim_paras["choices"][choice]["max"] = max_

    return optim_paras


def _parse_shocks(optim_paras, params):
    """Parse the shock parameters and create the Cholesky factor."""
    if sum(f"shocks_{i}" in params.index for i in ["sdcorr", "cov", "chol"]) >= 2:
        raise ValueError("It is not allowed to define multiple shock matrices.")
    elif "shocks_sdcorr" in params.index:
        sorted_shocks = _sort_shocks_sdcorr(optim_paras, params.loc["shocks_sdcorr"])
        cov = sdcorr_params_to_matrix(sorted_shocks)
        optim_paras["shocks_cholesky"] = robust_cholesky(cov)
    elif "shocks_cov" in params.index:
        sorted_shocks = _sort_shocks_cov_chol(
            optim_paras, params.loc["shocks_cov"], "cov"
        )
        cov = cov_params_to_matrix(sorted_shocks)
        optim_paras["shocks_cholesky"] = robust_cholesky(cov)
    elif "shocks_chol" in params.index:
        sorted_shocks = _sort_shocks_cov_chol(
            optim_paras, params.loc["shocks_chol"], "chol"
        )
        optim_paras["shocks_cholesky"] = chol_params_to_lower_triangular_matrix(
            sorted_shocks
        )
    else:
        raise NotImplementedError

    return optim_paras


def _sort_shocks_sdcorr(optim_paras, params):
    """Sort shocks of the standard deviation/correlation matrix.

    To fit :func:`estimagic.optimization.utilities.sdcorr_params_to_matrix`, standard
    deviations have to precede the elements of the remaining lower triangular matrix.

    """
    sds_flat = []
    corrs_flat = []

    for i, c_1 in enumerate(optim_paras["choices"]):
        for c_2 in list(optim_paras["choices"])[: i + 1]:
            if c_1 == c_2:
                sds_flat.append(params.loc[f"sd_{c_1}"])
            else:
                # The order in which choices are mentioned in the labels is not clear.
                # Check both combinations.
                if f"corr_{c_1}_{c_2}" in params.index:
                    corrs_flat.append(params.loc[f"corr_{c_1}_{c_2}"])
                elif f"corr_{c_2}_{c_1}" in params.index:
                    corrs_flat.append(params.loc[f"corr_{c_2}_{c_1}"])
                else:
                    raise ValueError(
                        f"Shock matrix has no entry for choice {c_1} and {c_2}"
                    )

    return sds_flat + corrs_flat


def _sort_shocks_cov_chol(optim_paras, params, type_):
    """Sort shocks of the covariance matrix or the Cholesky factor.

    To fit :func:`estimagic.optimization.utilities.cov_params_to_matrix` and
    :func:`estimagic.optimization.utilties.chol_params_to_lower_triangular_matrix`
    shocks have to be sorted like the lower triangular matrix or
    :func:`np.tril_indices(dim)`.

    """
    lower_triangular_flat = []

    for i, c_1 in enumerate(optim_paras["choices"]):
        for c_2 in list(optim_paras["choices"])[: i + 1]:
            if c_1 == c_2:
                label = "var" if type_ == "cov" else "chol"
                lower_triangular_flat.append(params.loc[f"{label}_{c_1}"])
            else:
                label = "cov" if type_ == "cov" else "chol"
                # The order in which choices are mentioned in the labels is not clear.
                # Check both combinations.
                if f"{label}_{c_1}_{c_2}" in params.index:
                    lower_triangular_flat.append(params.loc[f"{label}_{c_1}_{c_2}"])
                elif f"{label}_{c_2}_{c_1}" in params.index:
                    lower_triangular_flat.append(params.loc[f"{label}_{c_2}_{c_1}"])
                else:
                    raise ValueError(
                        f"Shock matrix has no entry for choice {c_1} and {c_2}"
                    )

    return lower_triangular_flat


def _parse_measurement_errors(optim_paras, params):
    """Parse correctly sorted measurement errors.

    optim_paras["is_meas_error"] is only False if there are no meas_error sds in params,
    not if they are all zero. Otherwise we would introduce a kink into the likelihood
    function.

    """
    meas_error = np.zeros(len(optim_paras["choices"]))

    if "meas_error" in params.index:
        optim_paras["is_meas_error"] = True
        labels = [f"sd_{choice}" for choice in optim_paras["choices_w_wage"]]
        assert set(params.loc["meas_error"].index) == set(labels), (
            "Standard deviations of measurement error have to be provided for all or "
            "none of the choices with wages. There can't be standard deviations of "
            "measurement errors for choices without wage."
        )
        meas_error[: len(labels)] = params.loc["meas_error"].loc[labels].to_numpy()
    else:
        optim_paras["is_meas_error"] = False

    optim_paras["meas_error"] = meas_error

    return optim_paras


def _parse_types(optim_paras, params):
    """Parse type shifts and type parameters.

    It is not explicitly enforced that all types have the same covariates, but it is
    implicitly enforced that the parameters form a valid matrix.

    """
    n_choices = len(optim_paras["choices"])

    if "type_shift" in params.index:
        n_types = _infer_number_of_types(params)
        types = [f"type_{i}" for i in range(2, n_types + 1)]
        optim_paras["type_covariates"] = (
            params.loc[types[0]].sort_index().index.to_list()
        )
        n_type_covariates = len(optim_paras["type_covariates"])

        optim_paras["type_prob"] = np.vstack(
            (
                np.zeros(n_type_covariates),
                params.loc[types]
                .sort_index()
                .to_numpy()
                .reshape(n_types - 1, n_type_covariates),
            )
        )

        type_shifts = np.zeros((n_types, n_choices))
        for type_ in range(2, n_types + 1):
            for i, choice in enumerate(optim_paras["choices"]):
                type_shifts[type_ - 1, i] = params.loc[
                    ("type_shift", f"type_{type_}_in_{choice}")
                ]
        optim_paras["type_shift"] = type_shifts
    else:
        optim_paras["type_shift"] = np.zeros((1, n_choices))

    optim_paras["n_types"] = optim_paras["type_shift"].shape[0]

    return optim_paras


def _infer_number_of_types(params):
    """Infer the number of types from parameters.

    Examples
    --------
    >>> params = pd.DataFrame(index=["type_3", "type_2"])
    >>> _infer_number_of_types(params)
    3
    >>> params = pd.DataFrame(index=["type_2", "type_3_asds", "type_423_"])
    >>> _infer_number_of_types(params)
    2

    """
    return (
        params.index.get_level_values(0)
        .str.extract(r"(\btype_[0-9]+\b)", expand=False)
        .nunique()
        + 1
    )


def _infer_choices_with_experience(params, options):
    """Infer choices with experiences.

    Example
    -------
    >>> options = {"covariates": {"a": "exp_white_collar + exp_a", "b": "exp_b >= 2"}}
    >>> index = pd.MultiIndex.from_product([["category"], ["a", "b"]])
    >>> params = pd.Series(index=index)
    >>> _infer_choices_with_experience(params, options)
    ['a', 'b', 'white_collar']

    """
    covariates = options["covariates"]
    parameters = params.index.get_level_values(1)

    used_covariates = [cov for cov in covariates if cov in parameters]

    matches = []
    for param in parameters:
        matches += re.findall(r"\bexp_([A-Za-z_]+)\b", str(param))
    for cov in used_covariates:
        matches += re.findall(r"\bexp_([A-Za-z_]+)\b", covariates[cov])

    return sorted(set(matches))


def _infer_choices_with_prefix(params, prefix):
    """Infer choices with prefix.

    Example
    -------
    >>> params = pd.Series(index=["wage_b", "wage_white_collar", "wage_a", "nonpec_c"])
    >>> _infer_choices_with_prefix(params, "wage")
    ['a', 'b', 'white_collar']

    """
    return sorted(
        params.index.get_level_values(0)
        .str.extract(fr"\b{prefix}_([A-Za-z_]+)\b", expand=False)
        .dropna()
        .unique()
    )


def _parse_lagged_choices(optim_paras, options, params):
    """Parse lagged choices from covariates and params.

    Lagged choices can only influence behavior of individuals through covariates of the
    utility function. Thus, check the covariates for any patterns like
    ``"lagged_choice_[0-9]+"``.

    Then, compare the number of lags required by covariates with the information on
    lagged choices in the parameter specification. For the estimation, there does not
    have to be any information on lagged choices. For the simulation, we need parameters
    to define the probability of a choice being the lagged choice.

    Warning
    -------
    UserWarning
        If not enough lagged choices are specified in params and the model can only be
        used for estimation.
    UserWarning
        If the model contains superfluous definitions of lagged choices.

    Example
    -------
    >>> optim_paras = {}
    >>> options = {
    ...     "covariates": {"covariate": "lagged_choice_2 + lagged_choice_1"},
    ...     "core_state_space_filters": [],
    ... }
    >>> index = pd.MultiIndex.from_tuples([("name", "covariate")])
    >>> params = pd.DataFrame(index=index)
    >>> _parse_lagged_choices(optim_paras, options, params)
    {'n_lagged_choices': 2}

    """
    regex_pattern = r"lagged_choice_([0-9]+)"

    # First, infer the number of lags from all used covariates.
    covariates = options["covariates"]
    parameters = params.index.get_level_values(1)
    used_covariates = [cov for cov in covariates if cov in parameters]

    matches = []
    for cov in used_covariates:
        matches += re.findall(regex_pattern, covariates[cov])

    n_lagged_choices = 0 if not matches else pd.to_numeric(matches).max()

    # Second, infer the number of lags defined in params.
    matches_params = list(
        params.index.get_level_values(0)
        .str.extract(regex_pattern, expand=False)
        .dropna()
        .unique()
    )

    lc_params = np.zeros(1) if not matches_params else pd.to_numeric(matches_params)
    n_lc_params = lc_params.max()
    undefined_lags = set(range(1, n_lagged_choices + 1)) - set(lc_params)

    # Check whether there is a discrepancy between the maximum number of lags specified
    # in covariates and filters or params.
    if n_lagged_choices > n_lc_params or undefined_lags:
        warnings.warn(
            "The distribution of initial lagged choices is insufficiently specified in "
            "params. This model cannot be used for simulation, only for estimation.",
            category=UserWarning,
        )
    elif n_lagged_choices < n_lc_params:
        warnings.warn(
            "The model contains superfluous information on lagged choices. The "
            f"covariates and filters require {n_lagged_choices} lagged choices whereas "
            f"{n_lc_params} lags are specified in params. Ignore superfluous lags in "
            "params.",
            category=UserWarning,
        )
    else:
        pass

    optim_paras["n_lagged_choices"] = n_lagged_choices

    # Add existing lagged choice parameters to ``optim_paras``.
    for match in (
        params.filter(like="lagged_choice_", axis=0).index.get_level_values(0).unique()
    ):
        optim_paras[match] = params.loc[match]

    return optim_paras
