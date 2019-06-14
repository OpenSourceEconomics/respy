import itertools

import numpy as np
import pandas as pd

from respy.config import HUGE_FLOAT
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import create_base_draws


class StateSpace:
    """The state space of a discrete choice dynamic programming model.

    Parameters
    ----------
    params : pd.Series or pd.DataFrame
        Contains parameters affected by optimization.
    options : dict
        Dictionary containing optimization independent model options.

    Attributes
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        lagged_choice and type information.
    indexer : np.ndarray
        Array with shape (n_periods, n_periods, n_periods, edu_max, n_choices, n_types).
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates of each state necessary
        to calculate rewards.
    wages : np.ndarray
        Array with shape (n_states_in_period, n_choices) which contains zeros in places
        for choices without wages.
    nonpec : np.ndarray
        Array with shape (n_states_in_period, n_choices).
    continuation_values : np.ndarray
        Array with shape (n_states, n_choices + 3) containing containing the emax of
        each choice of the subsequent period and the simulated or interpolated maximum
        of the current period.
    emax_value_functions : np.ndarray
        Array with shape (n_states, 1) containing the expected maximum of
        choice-specific value functions.

    """

    def __init__(self, params, options):
        params, optim_paras, options = process_params_and_options(params, options)

        # Add some arguments to the state space.
        n_periods = options["n_periods"]
        n_types = options["n_types"]
        self.seed = options["solution_seed"]

        self.base_draws_sol = create_base_draws(
            (n_periods, options["solution_draws"], len(options["sectors"])), self.seed
        )

        states_df, self.indexer = _create_state_space(options, n_types)

        _states_df = states_df.copy()
        _states_df.lagged_choice = _states_df.lagged_choice.cat.codes
        self.states = _states_df.to_numpy()

        base_covariates_df = _create_base_covariates(states_df, options["covariates"])

        self.covariates = _create_choice_covariates(
            base_covariates_df, states_df, params, options
        )

        self.wages, self.nonpec = _create_reward_components(
            self.states[:, -1], self.covariates, optim_paras, options
        )

        self.is_inadmissible = _create_is_inadmissible_indicator(states_df, options)

        self._create_slices_by_periods(n_periods)

    @property
    def states_per_period(self):
        """Get a list of states per period starting from the first period."""
        return np.array([len(range(i.start, i.stop)) for i in self.slices_by_periods])

    @property
    def num_states(self):
        """Get the total number states in the state space."""
        return self.states.shape[0]

    def update_systematic_rewards(self, optim_paras, options):
        self.wages, self.nonpec = _create_reward_components(
            self.states[:, -1], self.covariates, optim_paras, options
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

    def _create_slices_by_periods(self, n_periods):
        """Create slices to index all attributes in a given period.

        It is important that the returned objects are not fancy indices. Fancy indexing
        results in copies of array which decrease performance and raise memory usage.

        """
        self.slices_by_periods = []
        for i in range(n_periods):
            idx_start, idx_end = np.where(self.states[:, 0] == i)[0][[0, -1]]
            self.slices_by_periods.append(slice(idx_start, idx_end + 1))


def _create_state_space(options, n_types):
    """Create the state space.

    This process involves two steps. First, the core state space is created which
    abstracts from different initial experiences and instead uses the minimum initial
    experience per choice.

    Secondly, the state space is adjusted by all combinations of initial experiences and
    also filtered, excluding invalid states.

    """
    data = _create_core_state_space(options, n_types)

    exp_cols = [f"exp_{sector}" for sector in options["choices_w_exp"]]

    df = pd.DataFrame.from_records(
        data, columns=["period"] + exp_cols + ["lagged_choice", "type"]
    )

    codes_to_choices = {i: sec for i, sec in enumerate(options["choices"])}
    df.lagged_choice = df.lagged_choice.astype("category").cat.rename_categories(
        codes_to_choices
    )

    df = _adjust_state_space_with_different_starting_experiences(df, options)

    df = df.sort_values("period").reset_index(drop=True)

    indexer = _create_indexer(df, n_types, options)

    return df, indexer


def _create_core_state_space(options, n_types):
    """Create the core state space.

    The core state space abstracts from initial experiences and uses the maximum range
    between initial experiences and maximum experiences.

    """
    minimal_initial_experience = np.array(
        [np.min(options["sectors"][sec]["start"]) for sec in options["choices_w_exp"]]
    )
    additional_exp = options["maximum_exp"] - minimal_initial_experience

    data = []
    for period in range(options["n_periods"]):
        data += list(
            _create_core_state_space_per_period(
                period, additional_exp, n_types, options
            )
        )

    return data


def _create_core_state_space_per_period(
    period, additional_exp, n_types, options, experiences=None, pos=0
):
    """Create core state space per period.

    First, this function returns a state combined with all possible lagged choices and
    types.

    Secondly, if there exists a choice with experience in ``additional_exp[pos]``, loop
    over all admissible experiences, update the state and pass it to the same function,
    but moving to the next choice which accumulates experience.

    """
    experiences = experiences if experiences is not None else [0] * len(additional_exp)

    # Return experiences combined with lagged choices and types.
    for type_ in range(n_types):
        for choice in options["choices"]:
            yield [period] + experiences + [choice, type_]

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
                period, additional_exp, n_types, options, updated_experiences, pos + 1
            )


def _adjust_state_space_with_different_starting_experiences(df, options):
    # Create combinations of starting values
    initial_experiences_combinations = itertools.product(
        *[options["sectors"][sec]["start"] for sec in options["choices_w_exp"]]
    )

    exp_cols = df.filter(like="exp_").columns.tolist()

    container = []

    for initial_exp in initial_experiences_combinations:
        df_ = df.copy()

        # Add initial experiences.
        df_[exp_cols] += initial_exp

        # Check that max_experience is still fulfilled.
        df_ = df_.loc[
            df_[exp_cols].le(options["maximum_exp"]).all(axis="columns")
        ].copy()

        df_ = _filter_state_space(df_, options)

        container.append(df_)

    df = pd.concat(container, axis="rows", sort=False).drop_duplicates()

    return df


def _filter_state_space(df, options):
    col = "_restriction_{}"

    # Restriction counter
    counter = itertools.count()

    for definition in options["state_space_filters"]:

        # If "{i}" is in definition, loop over choices with experiences.
        if "{i}" in definition:
            for i in options["choices_w_exp"]:
                df[col.format(next(counter))] = df.eval(definition.format(i=i))

        # If "{j}" is in definition, loop over choices without experiences.
        elif "{j}" in definition:
            for j in options["choices_wo_exp"]:
                df[col.format(next(counter))] = df.eval(definition.format(j=j))

        else:
            df[col.format(next(counter))] = df.eval(definition)

    df = df[~df.filter(like="_restriction_").any(axis="columns")]
    df = df.drop(columns=df.filter(like="_restriction").columns.tolist())

    return df


def _create_indexer(df, n_types, options):
    n_exp_choices = len(options["choices_w_exp"])
    n_nonexp_choices = len(options["choices_wo_exp"])

    shape = (
        (options["n_periods"],)
        + tuple(options["maximum_exp"] + 1)
        + (n_exp_choices + n_nonexp_choices, n_types)
    )
    indexer = np.full(shape, -1)

    idx = (
        (df.period,)
        + tuple(df[f"exp_{i}"] for i in options["choices_w_exp"])
        + (df.lagged_choice.cat.codes, df.type)
    )
    indexer[idx] = np.arange(df.shape[0])

    return indexer


def _create_reward_components(types, covariates, optim_paras, options):
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
    wage_labels = [f"wage_{sec}" for sec in options["choices_w_wage"]]
    log_wages = np.column_stack(
        [np.dot(covariates[w], optim_paras[w]) for w in wage_labels]
    )

    nonpec_labels = [f"nonpec_{sec}" for sec in options["sectors"]]
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


def _create_choice_covariates(covariates_df, states_df, params, options):
    """Create the covariates for each choice.

    Parameters
    ----------
    covariates_df : pd.DataFrame
        DataFrame with the basic covariates.
    states_df : pd.DataFrame
        DataFrame with the state information.
    params : pd.DataFrame or pd.Series
        The parameter specification.
    options : dict
        Contains model options.

    Returns
    -------
    covariates : dict
        Dictionary where values are the wage or non-pecuniary covariates for choices.

    """
    all_data = pd.concat([covariates_df, states_df], axis="columns", sort=False)

    covariates = {}

    for choice in options["sectors"]:
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
    states : pd.DataFrame
        DataFrame with shape (n_states, n_choices_w_exp + 3) containing period,
        experiences, choice_lagged and type of each state.
    covariates_spec : dict
        Keys represent covariates and values are strings passed to ``df.eval``.

    Returns
    -------
    covariates : pd.DataFrame
        DataFrame with shape (n_states, n_covariates).

    """
    covariates = states.copy()

    for covariate, definition in covariates_spec.items():
        covariates[covariate] = covariates.eval(definition)

    covariates = covariates.drop(columns=states.columns).astype(float)

    return covariates


def _create_is_inadmissible_indicator(states, options):
    # Make maximum experience for each sector available as local variable.
    locals_ = {
        f"max_exp_{sec}": options["sectors"][sec]["max"]
        for sec in options["choices_w_exp"]
    }

    df = states.copy()

    for column, definition in options["inadmissible_states"].items():
        df[column] = df.eval(definition, local_dict=locals_)

    is_inadmissible = df[options["choices"]].to_numpy()

    return is_inadmissible
