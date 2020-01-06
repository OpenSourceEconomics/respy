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
from respy.config import MAX_FLOAT
from respy.config import SEED_STARTUP_ITERATION_GAP
from respy.pre_processing.model_checking import validate_options
from respy.shared import normalize_probabilities

warnings.simplefilter("error", category=pd.errors.PerformanceWarning)


def process_params_and_options(params, options):
    """Process ``params`` and ``options``.

    This function is interface for parsing the model specification given by the user.

    """
    options = _read_options(options)
    options = {**DEFAULT_OPTIONS, **options}
    options = _create_internal_seeds_from_user_seeds(options)
    options = _identify_relevant_covariates(options, params)
    validate_options(options)

    params = _read_params(params)
    optim_paras = _parse_parameters(params, options)

    optim_paras, options = _sync_optim_paras_and_options(optim_paras, options)

    return optim_paras, options


def _read_options(input_):
    """Read the options which can either be a dictionary or a path."""
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
    """Read the parameters which can either be a path, a Series, or a DataFrame."""
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
    optim_paras = {"delta": params.loc[("delta", "delta")]}

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
    """Parse observed variables and their levels."""
    optim_paras["observables"] = {}

    regex_observables = r"\bobservable_([a-z0-9_]+)_[0-9a-z]+\b"
    observable_counts = (
        params.index.get_level_values("category")
        .str.extract(regex_observables, expand=False)
        .value_counts()
        .sort_index()
    )

    for observable in observable_counts.index:
        regex_pattern = fr"\bobservable_{observable}_([0-9a-z]+)\b"
        parsed_parameters = _parse_probabilities_or_logit_coefficients(
            params, regex_pattern
        )
        optim_paras["observables"][observable] = parsed_parameters

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
    """Process initial experience distributions and maximum experience."""
    for choice in optim_paras["choices_w_exp"]:
        regex_for_levels = fr"\binitial_exp_{choice}_([0-9]+)\b"
        parsed_parameters = _parse_probabilities_or_logit_coefficients(
            params, regex_for_levels
        )
        if parsed_parameters is None:
            parsed_parameters = {0: pd.Series(index=["constant"], data=0)}
        optim_paras["choices"][choice]["start"] = parsed_parameters

        default_max = max(parsed_parameters) + options["n_periods"] - 1
        max_ = int(params.get(("maximum_exp", choice), default_max))
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
    if "type_0" in params.index.get_level_values("category"):
        raise ValueError(
            "'type_0' cannot be used to specify the probability mass function of types "
            "as it has to be zero such that all parameters are identified."
        )
    if "type_0" in params.index.get_level_values("name"):
        raise ValueError(
            "'type_0' cannot be used as a utility covariate as it must be zero due to "
            "normalization. All other types are expressed in relation to the first."
        )

    n_types = _infer_number_of_types(params)

    if n_types >= 2:
        # Parse type probabilities.
        parsed_parameters = _parse_probabilities_or_logit_coefficients(
            params, r"\btype_([0-9]+)\b"
        )
        parsed_parameters = {k: v for k, v in parsed_parameters.items()}
        default = {i: pd.Series(data=[0], index=["constant"]) for i in range(n_types)}
        optim_paras["type_prob"] = {**default, **parsed_parameters}

        # Parse type covariates which makes estimation via maximum likelihood faster.
        optim_paras["type_covariates"] = {
            cov
            for level_dict in optim_paras["type_prob"].values()
            for cov in level_dict.index
        }

    optim_paras["n_types"] = n_types

    return optim_paras


def _infer_number_of_types(params):
    """Infer the number of types from parameters which is zero by default.

    Examples
    --------
    An example without types:

    >>> tuples = [("wage_a", "constant"), ("nonpec_edu", "exp_edu")]
    >>> index = pd.MultiIndex.from_tuples(tuples, names=["category", "name"])
    >>> s = pd.Series(index=index)
    >>> _infer_number_of_types(s)
    1

    And one with types:

    >>> tuples = [("wage_a", "type_3"), ("nonpec_edu", "type_2")]
    >>> index = pd.MultiIndex.from_tuples(tuples, names=["category", "name"])
    >>> s = pd.Series(index=index)
    >>> _infer_number_of_types(s)
    4

    """
    n_types = (
        params.index.get_level_values("name")
        .str.extract(r"\btype_([0-9]+)\b", expand=False)
        .fillna(0)
        .astype(int)
        .max()
        + 1
    )

    return n_types


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

    """
    regex_pattern = r"lagged_choice_([0-9]+)"

    # First, infer the number of lags from all covariates.
    covariates = options["covariates"]
    matches = []
    for cov in covariates:
        matches += re.findall(regex_pattern, covariates[cov])

    n_lc_covariates = 0 if not matches else pd.to_numeric(matches).max()

    # Second, infer the number of lags defined in params.
    matches_params = list(
        params.index.get_level_values("category")
        .str.extract(regex_pattern, expand=False)
        .dropna()
        .unique()
    )

    lc_params = [0] if not matches_params else pd.to_numeric(matches_params)
    n_lc_params = max(lc_params)

    # Todo: I am not sure whether we want to emit a warning? Defaults are fine.

    # Check whether there is a discrepancy between the maximum number of lags specified
    # in covariates or params.
    if n_lc_covariates > n_lc_params:
        warnings.warn(
            "The distribution of initial lagged choices is insufficiently specified in "
            f"the parameters. Covariates require {n_lc_covariates} lagged choices and "
            f"parameters define {n_lc_params}. Missing lags have equiprobable choices.",
            category=UserWarning,
        )

    elif n_lc_covariates < n_lc_params:
        warnings.warn(
            "The parameters contain superfluous information on lagged choices. The "
            f"covariates require {n_lc_covariates} lags whereas parameters provide "
            f"information on {n_lc_params} lags. Superfluous lags are ignored.",
            category=UserWarning,
        )

    else:
        pass

    optim_paras["n_lagged_choices"] = n_lc_covariates

    # Add existing lagged choice parameters to ``optim_paras``.
    for lag in range(1, n_lc_covariates + 1):
        parsed_parameters = _parse_probabilities_or_logit_coefficients(
            params, fr"lagged_choice_{lag}_([A-Za-z_]+)"
        )

        # If there are no parameters for the specific lag, assume equiprobable choices.
        if parsed_parameters is None:
            parsed_parameters = {
                choice: pd.Series(index=["constant"], data=0)
                for choice in optim_paras["choices"]
            }

        # If there are parameters, put zero probability on missing choices.
        else:
            defaults = {
                choice: pd.Series(index=["constant"], data=-MAX_FLOAT)
                for choice in optim_paras["choices"]
            }
            parsed_parameters = {**defaults, **parsed_parameters}

        optim_paras[f"lagged_choice_{lag}"] = parsed_parameters

    return optim_paras


def _parse_probabilities_or_logit_coefficients(params, regex_for_levels):
    r"""Parse probabilities or logit coefficients of parameter groups.

    Some parameters form a group to specify a distribution. The parameters can either be
    probabilities from a probability mass function. For example, see the specification
    of initial years of schooling in the extended model of Keane and Wolpin (1997).

    On the other hand, parameters and their corresponding covariates can form the inputs
    of a :func:`scipy.specical.softmax` which generates the probability mass function.
    This distribution can be more complex.

    Internally, probabilities are also converted to logit coefficients to align the
    interfaces. To convert probabilities to the appropriate multinomial logit (softmax)
    coefficients, use a constant for covariates and note that the sum in the denominator
    is equal for all probabilities and, thus, can be treated as a constant. The
    following formula shows that the multinomial coefficients which produce the same
    probability mass function are equal to the logs of probabilities.

    .. math::

        p_i      &= \frac{e^{x_i \beta_i}}{\sum_j e^{x_j \beta_j}} \
                 &= \frac{e^{\beta_i}}{\sum_j e^{\beta_j}} \
        log(p_i) &= \beta_i - \log(\sum_j e^{\beta_j}) \
                 &= \beta_i - C

    Raises
    ------
    ValueError
        If probabilities and multinomial logit coefficients are mixed.

    Warnings
    --------
    The user is warned if the discrete probabilities of a probability mass function do
    not sum to one.

    """
    mask = (
        params.index.get_level_values("category")
        .str.extract(regex_for_levels, expand=False)
        .notna()
    )
    n_parameters = mask.sum()

    # If parameters for initial experiences are specified, the parameters can either
    # be probabilities or multinomial logit coefficients.
    if n_parameters:
        # Work on subset.
        sub = params.loc[mask].copy()

        levels = sub.index.get_level_values("category").str.extract(
            regex_for_levels, expand=False
        )
        levels = pd.to_numeric(levels, errors="ignore")
        unique_levels = sorted(levels.unique())

        n_probabilities = (sub.index.get_level_values("name") == "probability").sum()

        # It is allowed to specify the shares of initial experiences as
        # probabilities. Then, the probabilities are replaced with their logs to
        # recover the probabilities with a multinomial logit model.
        if n_probabilities == len(unique_levels) == n_parameters:
            if sub.sum() != 1:
                warnings.warn(
                    f"The probabilities for parameter group {regex_for_levels} do not "
                    "sum to one.",
                    category=UserWarning,
                )
                sub = normalize_probabilities(sub)

            # Clip at the smallest representable number to prevent -infinity for log(0).
            sub = np.log(np.clip(sub, 1 / MAX_FLOAT, None))
            sub = sub.rename(index={"probability": "constant"}, level="name")

        elif n_probabilities > 0:
            raise ValueError(
                "Cannot mix probabilities and multinomial logit coefficients for the "
                f"parameter group: {regex_for_levels}."
            )

        # Drop level 'category' from :class:`pd.MultiIndex`.
        s = sub.droplevel(axis="index", level="category")
        # Insert parameters for every level of initial experiences.
        container = {level: s.loc[levels == level] for level in unique_levels}

    # If no parameters are provided, return `None` so that the default is handled
    # outside the function.
    else:
        container = None

    return container


def _identify_relevant_covariates(options, params):
    """Identify the relevant covariates.

    We try to make every model as sparse as possible which means discarding covariates
    which are irrelevant. The immediate benefit is that memory consumption and start-up
    costs are reduced.

    An advantage further downstream is that the number of lagged choices is inferred
    from covariates. Eliminating irrelevant covariates might reduce the number of
    implemented lags.

    """
    covariates = options["covariates"]

    relevant_covariates = {}
    for cov in covariates:
        if cov in params.index.get_level_values("name"):
            relevant_covariates[cov] = covariates[cov]

    n_relevant_covariates_changed = True
    while n_relevant_covariates_changed:
        n_relevant_covariates = len(relevant_covariates)

        for cov in covariates:
            for relevant_cov in list(relevant_covariates):
                if cov in relevant_covariates[relevant_cov]:
                    # Append the covariate to the front such that nested covariates are
                    # created in the beginning.
                    relevant_covariates = {cov: covariates[cov], **relevant_covariates}

        if n_relevant_covariates == len(relevant_covariates):
            n_relevant_covariates_changed = False
        else:
            n_relevant_covariates_changed = True

    options["covariates"] = relevant_covariates

    return options


def _sync_optim_paras_and_options(optim_paras, options):
    """Sync ``optim_paras`` and ``options`` after they have been parsed separately."""
    optim_paras["n_periods"] = options["n_periods"]

    # Create covariates for the reward functions.
    if optim_paras["n_types"] >= 2:
        type_covariates = {
            f"type_{i}": f"type == {i}" for i in range(1, optim_paras["n_types"])
        }

        options["covariates"] = {**options["covariates"], **type_covariates}

    options = _convert_labels_in_formulas_to_codes(options, optim_paras)

    return optim_paras, options


def _convert_labels_in_formulas_to_codes(options, optim_paras):
    """Convert labels in covariates, filters and inadmissible formulas to codes.

    Characteristics with labels are either choices or observables. Choices are ordered
    as in ``optim_paras["choices"]`` and observables alphabetically.

    Labels can either be in single or double quote strings which has to be checked.

    """
    for covariate, formula in options["covariates"].items():
        options["covariates"][covariate] = _replace_choices_and_observables_in_formula(
            formula, optim_paras
        )

    options = _convert_labels_in_filters_to_codes(optim_paras, options)

    for choice in optim_paras["choices"]:
        for i, formula in enumerate(options["inadmissible_states"].get(choice, [])):
            options["inadmissible_states"][choice][
                i
            ] = _replace_choices_and_observables_in_formula(formula, optim_paras)

    return options


def _replace_in_single_or_double_quotes(val, from_, to):
    """Replace a value in a string enclosed in single or double quotes."""
    return val.replace(f"'{from_}'", f"{to}").replace(f'"{from_}"', f"{to}")


def _replace_choices_and_observables_in_formula(formula, optim_paras):
    """Replace choices and observables in formula.

    Choices and levels of an observable can have string identifier which are replaced
    with their codes.

    """
    observables = optim_paras["observables"]

    for i, choice in enumerate(optim_paras["choices"]):
        formula = _replace_in_single_or_double_quotes(formula, choice, i)

    for observable in observables:
        for i, obs in enumerate(observables[observable]):
            if isinstance(obs, str):
                formula = _replace_in_single_or_double_quotes(formula, obs, i)

    return formula


def _convert_labels_in_filters_to_codes(optim_paras, options):
    """Convert labels in ``"core_state_space_filters"`` to codes.

    The filters are used to remove states from the state space which are inadmissible
    anyway.

    A filter might look like this::

        "lagged_choice_1 == '{k}' and exp_{k} == 0"

    ``{k}`` is replaced by the actual choice name whereas ``'{k}'`` or ``"{k}"`` is
    replaced with the internal choice code.

    """
    filters = []

    for filter_ in options["core_state_space_filters"]:
        # If "{i}" is in filter_, loop over choices with experiences.
        if "{i}" in filter_:
            for choice in optim_paras["choices_w_exp"]:
                fltr_ = filter_.replace("{i}", choice)
                fltr_ = _replace_choices_and_observables_in_formula(fltr_, optim_paras)
                filters.append(fltr_)

        # If "{j}" is in filter_, loop over choices without experiences.
        elif "{j}" in filter_:
            for choice in optim_paras["choices_wo_exp"]:
                fltr = filter_.replace("{j}", choice)
                fltr = _replace_choices_and_observables_in_formula(fltr, optim_paras)
                filters.append(fltr)

        # If "{k}" is in filter_, loop over choices with wage.
        elif "{k}" in filter_:
            for choice in optim_paras["choices_w_wage"]:
                fltr = filter_.replace("{k}", choice)
                fltr = _replace_choices_and_observables_in_formula(fltr, optim_paras)
                filters.append(fltr)

        else:
            filter_ = _replace_choices_and_observables_in_formula(filter_, optim_paras)
            filters.append(filter_)

    options["core_state_space_filters"] = filters

    return options
