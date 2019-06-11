import itertools

import numpy as np
import pandas as pd

from respy.config import HUGE_FLOAT
from respy.pre_processing.model_processing import process_options
from respy.pre_processing.model_processing import process_params
from respy.shared import create_base_draws


class StateSpace:
    """Class containing all objects related to the state space of a discrete choice
    dynamic programming model.

    Parameters
    ----------
    attr : dict
        Dictionary containing model attributes.
    optim_paras : dict
        Dictionary containing parameters affected by optimization.

    Attributes
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        choice_lagged and type information.
    indexer : np.ndarray
        Array with shape (num_periods, num_periods, num_periods, edu_max, n_choices,
        num_types).
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates of each state necessary
        to calculate rewards.
    wages : np.ndarray
        Array with shape (n_states_in_period, n_choices) which contains zeros in places
        for choices without wages.
    nonpec : np.ndarray
        Array with shape (n_states_in_period, n_choices).
    emaxs : np.ndarray
        Array with shape (num_states, 5) containing containing the emax of each choice
        (OCCUPATION A, OCCUPATION B, SCHOOL, HOME) of the subsequent period and the
        simulated or interpolated maximum of the current period.
    num_periods : int
        Number of periods.
    num_types : int
        Number of types.
    states_columns : list
        List of column names in ``self.states``.
    wages_columns : list
        List of column names in ``self.wages``
    nonpec_columns : list
        List of column names in ``self.nonpec``
    emaxs_columns : list
        List of column names in ``self.continuation_values``.

    """

    states_columns = ["period", "exp_a", "exp_b", "exp_edu", "choice_lagged", "type"]

    wages_columns = ["wage_a", "wage_b"]
    nonpec_columns = ["nonpec_a", "nonpec_b", "nonpec_edu", "nonpec_home"]
    emaxs_columns = ["emax_a", "emax_b", "emax_edu", "emax_home", "emax"]

    def __init__(self, params, options):
        params, optim_paras = process_params(params)
        options = process_options(options)

        # Add some arguments to the state space.
        self.edu_max = options["education_max"]
        self.num_periods = options["num_periods"]
        self.num_types = optim_paras["num_types"]
        self.seed = options["solution_seed"]

        self.base_draws_sol = create_base_draws(
            (self.num_periods, options["solution_draws"], 4), self.seed
        )

        # Create attributes for new state space.
        maximum_exp = np.array([self.num_periods, self.num_periods, self.edu_max])
        minimum_exp = np.array([0, 0, options["education_start"]])
        n_nonexp_choices = 1

        states_df, self.indexer = _create_state_space(
            self.num_periods,
            maximum_exp,
            minimum_exp,
            n_nonexp_choices,
            self.num_types,
            options["state_space_filters"],
        )

        self.states = states_df.to_numpy()

        base_covariates_df = _create_base_covariates(states_df, options["covariates"])

        self.covariates = _create_choice_covariates(
            base_covariates_df, states_df, params
        )

        self.wages, self.nonpec = _create_reward_components(
            self.states[:, 5], self.covariates, optim_paras
        )

        self.is_inadmissible = _create_is_inadmissible_indicator(states_df, options)

        self._create_slices_by_periods()

    @property
    def states_per_period(self):
        """Get a list of states per period starting from the first period."""
        return np.array([len(range(i.start, i.stop)) for i in self.slices_by_periods])

    @property
    def num_states(self):
        """Get the total number states in the state space."""
        return self.states.shape[0]

    def update_systematic_rewards(self, optim_paras):
        self.wages, self.nonpec = _create_reward_components(
            self.states[:, 5], self.covariates, optim_paras
        )

    def get_attribute_from_period(self, attr, period):
        """Get an attribute of the state space sliced to a given period.

        Parameters
        ----------
        attr : str
            String of attribute name, e.g. ``"states"``.
        period : int
            Attribute is retrieved from this period.

        """
        if attr == "covariates":
            raise AttributeError("Attribute covariates cannot be retrieved by periods.")
        else:
            pass

        try:
            attribute = getattr(self, attr)
        except AttributeError as e:
            raise AttributeError(f"StateSpace has no attribute {attr}.").with_traceback(
                e.__traceback__
            )

        try:
            indices = self.slices_by_periods[period]
        except IndexError as e:
            raise IndexError(f"StateSpace has no period {period}.").with_traceback(
                e.__traceback__
            )

        return attribute[indices]

    def _create_slices_by_periods(self):
        """Create slices to index all attributes in a given period.

        It is important that the returned objects are not fancy indices. Fancy indexing
        results in copies of array which decrease performance and raise memory usage.

        """
        self.slices_by_periods = []
        for i in range(self.num_periods):
            idx_start, idx_end = np.where(self.states[:, 0] == i)[0][[0, -1]]
            self.slices_by_periods.append(slice(idx_start, idx_end + 1))


def _create_state_space(
    n_periods, maximum_exp, minimum_exp, n_nonexp_choices, n_types, filters
):
    """Create the state space.

    This process involves two steps. First, the core state space is created which
    abstracts from different initial experiences and instead uses the minimum initial
    experience per choice.

    Secondly, the state space is adjusted by all combinations of initial experiences and
    also filtered, excluding invalid states.

    """
    df = _create_core_state_space(
        n_periods, maximum_exp, minimum_exp, n_nonexp_choices, n_types
    )

    df = _adjust_state_space_with_different_starting_experiences(
        df, minimum_exp, maximum_exp, n_nonexp_choices, filters
    )

    df = (
        df.sort_values("period")
        .reset_index(drop=True)
        .rename(
            columns={
                "exp_0": "exp_a",
                "exp_1": "exp_b",
                "exp_2": "exp_edu",
                "lagged_choice": "choice_lagged",
            }
        )
    )[["period", "exp_a", "exp_b", "exp_edu", "choice_lagged", "type"]]

    indexer = _create_indexer(df, n_periods, n_nonexp_choices, n_types, maximum_exp)

    return df, indexer


def _create_core_state_space(
    n_periods, maximum_exp, minimum_exp, n_nonexp_choices, n_types
):
    """Create the core state space.

    The core state space abstracts from initial experiences and uses the maximum range
    between initial experiences and maximum experiences.

    """
    min_min_exp = np.array([np.min(i) for i in minimum_exp])
    additional_exp = maximum_exp - np.array(min_min_exp)

    data = []
    for period in range(n_periods):
        data += list(
            _create_core_state_space_per_period(
                period, additional_exp, n_nonexp_choices, n_types
            )
        )

    df = pd.DataFrame.from_records(
        data, columns=["period", "exp_0", "exp_1", "exp_2", "lagged_choice", "type"]
    )

    return df


def _create_core_state_space_per_period(
    period, additional_exp, n_nonexp_choices, n_types, experiences=None, pos=0
):
    """Create core state space per period.

    First, this function returns a state combined with all possible lagged choices and
    types.

    Secondly, if there exists a choice with experience in ``additional_exp[pos]``, loop
    over all admissible experiences, update the state and pass it to the same function,
    but moving to the next choice which accumulates experience.

    """
    experiences = [0] * len(additional_exp) if experiences is None else experiences

    # Return experiences combined with lagged choices and types.
    for t in range(n_types):
        for i in range(len(experiences) + n_nonexp_choices):
            yield [period] + experiences + [i, t]

    # Check if there is an additional choice left to start another loop.
    if pos < len(experiences):
        # Upper bound of additional experience is given by the remaining time or the
        # maximum experience which can be accumulated in experience[pos].
        remaining_time = period - np.sum(experiences)
        max_experience = additional_exp[pos]

        # +1 is necessary so that the remaining time or max_experience is exhausted.
        for i in range(min(remaining_time, max_experience) + 1):
            # Update experiences and call the same function with the next choice.
            updated_experiences = experiences.copy()
            updated_experiences[pos] += i
            yield from _create_core_state_space_per_period(
                period,
                additional_exp,
                n_nonexp_choices,
                n_types,
                updated_experiences,
                pos + 1,
            )


def _adjust_state_space_with_different_starting_experiences(
    df, minimum_exp, maximum_exp, n_nonexp_choices, filters
):
    # TODO: Has to move to model processing.
    minimum_exp_ = [[i] if isinstance(i, int) else i for i in minimum_exp]
    # Create combinations of starting values
    starting_combinations = itertools.product(*minimum_exp_)

    exp_cols = df.filter(like="exp_").columns.tolist()

    container = []

    for start_comb in starting_combinations:
        df_ = df.copy()

        # Add initial experiences.
        df_[exp_cols] += start_comb

        # Check that time constraint is still fulfilled.
        sum_experiences = df_[exp_cols].subtract(start_comb).sum(axis="columns")
        df_ = df_.loc[df_.period.ge(sum_experiences)].copy()

        # Check that max_experience is still fulfilled.
        df_ = df_.loc[df_[exp_cols].le(maximum_exp).all(axis="columns")].copy()

        df_ = _filter_state_space(df_, filters, n_nonexp_choices, start_comb)

        container.append(df_)

    df = pd.concat(container, axis="rows", sort=False).drop_duplicates()

    return df


def _filter_state_space(df, filters, n_nonexp_choices, initial_exp):
    col = "_restriction_{}"
    n_exp_choices = len(initial_exp)

    # Restriction counter
    counter = itertools.count()

    for definition in filters:

        # If "{i}" is in definition, loop over choices with experiences.
        if "{i}" in definition:
            for i in range(n_exp_choices):
                df[col.format(next(counter))] = df.eval(definition.format(i=i))

        # If "{j}" is in definition, loop over choices without experiences.
        elif "{j}" in definition:
            for j in range(n_exp_choices, n_exp_choices + n_nonexp_choices):
                df[col.format(next(counter))] = df.eval(definition.format(j=j))

        else:
            df[col.format(next(counter))] = df.eval(definition)

    df = df[~df.filter(like="_restriction_").any(axis="columns")]
    df = df.drop(columns=df.filter(like="_restriction").columns.tolist())

    return df


def _create_indexer(df, n_periods, n_nonexp_choices, n_types, maximum_exp):
    n_exp_choices = maximum_exp.shape[0]
    maximum_exp[2] += 1

    shape = (
        (n_periods,) + tuple(maximum_exp) + (n_exp_choices + n_nonexp_choices, n_types)
    )
    indexer = np.full(shape, -1)

    indexer[
        df.period, df.exp_a, df.exp_b, df.exp_edu, df.choice_lagged, df.type
    ] = np.arange(df.shape[0])

    return indexer


def _create_reward_components(types, covariates, optim_paras):
    """Calculate systematic rewards for each state.

    Wages are only available for some choices, i.e. n_nonpec >= n_wages. We extend the
    array of wages with ones for the difference in dimensions, n_nonpec - n_wages. Ones
    are necessary as it facilitates the aggregation of reward components in
    :func:`calculate_emax_value_functions` and related functions.

    Parameters
    ----------
    types : np.ndarray
        Array with shape (n_states,) containing type information.
    covariates : dict
        Dictionary with covariate arrays for wage and nonpec rewards.
    optim_paras : dict
        Contains parameters affected by the optimization.

    """
    wage_labels = ["wage_a", "wage_b"]
    log_wages = np.column_stack(
        [np.dot(covariates[w], optim_paras[w]) for w in wage_labels]
    )

    nonpec_labels = ["nonpec_a", "nonpec_b", "nonpec_edu", "nonpec_home"]
    nonpec = np.column_stack(
        [np.dot(covariates[n], optim_paras[n]) for n in nonpec_labels]
    )

    type_deviations = optim_paras["type_shifts"][types]

    log_wages += type_deviations[:, :2]
    nonpec[:, 2:] += type_deviations[:, 2:]

    wages = np.clip(np.exp(log_wages), 0.0, HUGE_FLOAT)

    # Extend wages to dimension of non-pecuniary rewards.
    additional_dim = nonpec.shape[1] - log_wages.shape[1]
    wages = np.column_stack((wages, np.ones((wages.shape[0], additional_dim))))

    return wages, nonpec


def _create_choice_covariates(covariates_df, states_df, params):
    """Create the covariates for each choice.

    Parameters
    ----------
    covariates_df : pd.DataFrame
        DataFrame with the basic covariates.
    states_df : pd.DataFrame
        DataFrame with the state information.
    params : pd.DataFrame or pd.Series
        The parameter specification.

    Returns
    -------
    wage_covariates: list
        List of length ``n_choices`` with covariate arrays for systematic wages.
    nonpec_covariates: list
        List of length ``n_choices`` with covariate arrays for non-pecuniary rewards.

    """
    all_data = pd.concat([covariates_df, states_df], axis="columns", sort=False)

    covariates = {}

    for choice in ["a", "b", "edu", "home"]:
        if f"wage_{choice}" in params.index:
            wage_columns = params.loc[f"wage_{choice}"].index
            covariates[f"wage_{choice}"] = all_data[wage_columns].to_numpy()

        nonpec_columns = params.loc[f"nonpec_{choice}"].index
        covariates[f"nonpec_{choice}"] = all_data[nonpec_columns].to_numpy()

    for key, val in covariates.items():
        covariates[key] = np.ascontiguousarray(val)

    return covariates


def _create_base_covariates(states, covariates_spec):
    """Create set of covariates for each state.

    Parameters
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        choice_lagged and type of each state.
    covariates_spec : dict
        Dictionary where keys represent covariates and values are strings which can be
        passed to pd.eval.

    Returns
    -------
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates of each state.

    """
    covariates = states.copy()

    for covariate, definition in covariates_spec.items():
        covariates[covariate] = covariates.eval(definition)

    covariates = covariates.drop(columns=states.columns).astype(float)

    return covariates


def _create_is_inadmissible_indicator(states, options):
    df = states.copy()

    # Add maximum of experiences.
    for key in options.keys():
        if "max" in key:
            df[key] = options[key]
        else:
            pass

    for column, definition in options["inadmissible_states"].items():
        df[column] = df.eval(definition)

    is_inadmissible = df[options["inadmissible_states"].keys()].to_numpy()

    return is_inadmissible
