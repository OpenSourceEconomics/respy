import itertools

import numpy as np
import pandas as pd

from respy.config import HUGE_FLOAT
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import create_base_draws
from respy.shared import downcast_to_smallest_dtype


class StateSpace:
    """The state space of a discrete choice dynamic programming model.

    Parameters
    ----------
    params : pandas.Series or pandas.DataFrame
        Contains parameters affected by optimization.
    options : dict
        Dictionary containing optimization independent model options.

    Attributes
    ----------
    states : numpy.ndarray
        Array with shape (n_states, n_choices + 3) containing period, experiences,
        lagged_choice and type information.
    indexer : numpy.ndarray
        Array with shape (n_periods, n_periods, n_periods, edu_max, n_choices, n_types).
    covariates : numpy.ndarray
        Array with shape (n_states, n_covariates) containing covariates of each state
        necessary to calculate rewards.
    wages : numpy.ndarray
        Array with shape (n_states_in_period, n_choices) which contains zeros in places
        for choices without wages.
    nonpec : numpy.ndarray
        Array with shape (n_states_in_period, n_choices).
    continuation_values : numpy.ndarray
        Array with shape (n_states, n_choices + 3) containing containing the emax of
        each choice of the subsequent period and the simulated or interpolated maximum
        of the current period.
    emax_value_functions : numpy.ndarray
        Array with shape (n_states, 1) containing the expected maximum of
        choice-specific value functions.

    """

    def __init__(self, params, options):
        params, optim_paras, options = process_params_and_options(params, options)

        self.base_draws_sol = create_base_draws(
            (options["n_periods"], options["solution_draws"], len(options["choices"])),
            options["solution_seed"],
        )

        states_df, self.indexer = _create_state_space(options)

        _states_df = states_df.copy()
        _states_df.lagged_choice = _states_df.lagged_choice.cat.codes
        _states_df = _states_df.apply(downcast_to_smallest_dtype)
        self.states = _states_df.to_numpy()

        base_covariates_df = create_base_covariates(states_df, options["covariates"])
        base_covariates_df = base_covariates_df.apply(downcast_to_smallest_dtype)

        self.covariates = _create_choice_covariates(
            base_covariates_df, states_df, params, options
        )

        self.wages, self.nonpec = _create_reward_components(
            self.states[:, -1], self.covariates, optim_paras, options
        )

        self.is_inadmissible = _create_is_inadmissible_indicator(states_df, options)

        self._create_slices_by_periods(options["n_periods"])

    def update_systematic_rewards(self, optim_paras, options):
        """Update wages and non-pecuniary rewards.

        During the estimation, the rewards need to be updated according to the new
        parameters whereas the covariates stay the same.

        """
        self.wages, self.nonpec = _create_reward_components(
            self.states[:, -1], self.covariates, optim_paras, options
        )

    def get_attribute_from_period(self, attr, period):
        """Get an attribute of the state space sliced to a given period.

        Parameters
        ----------
        attr : str
            Attribute name, e.g. ``"states"`` to retrieve ``self.states``.
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


def _create_state_space(options):
    """Create the state space.

    The state space of the model are all feasible combinations of the period,
    experiences, lagged choices and types.

    Creating the state space involves two steps. First, the core state space is created
    which abstracts from levels of initial experiences and instead uses the minimum
    initial experience per choice.

    Secondly, the state space is adjusted by all combinations of initial experiences and
    also filtered, excluding invalid states.

    Notes
    -----
    Here are some details on the implementation.

    - In the process of creating this function, we came up with several different ideas.
      Basically, there two fringe cases to find all valid states in the state space.
      First, all combinations of state attributes are created. Then, only valid states
      are selected. The problem with this approach is that the state space is extremely
      sparse. The number of combinations created by using ``itertools.product`` or
      ``np.meshgrid`` is much higher than the number of valid states. Because of that,
      we ran into memory or runtime problems which seemed unsolvable.

      The second approach is more similar to the actual process were states are created
      by incrementing experiences from period to period. In an extreme case, a function
      mimics an agent in one period and recursively creates updates of itself in future
      periods. Using this approach, we ran into the Python recursion limit and runtime
      problems, but it might be feasible.

      These two approaches build the frame for thinking about a solution to this problem
      where filtering is, first, applied after creating a massive amount of candidate
      states, or, secondly, before creating states. A practical solution must take into
      account that some restrictions to the state space are more important than others
      and should be applied earlier. Others can be delayed.

      As a compromise, we built on the former approach in
      :func:`~respy.tests._former_code._create_state_space_kw94` which loops over
      choices and possible experience values. Thus, it incorporates some fundamental
      restrictions like time limits and needs less filtering.

    - The former implementation,
      :func:`~respy.tests._former_code._create_state_space_kw94`, had four hard-coded
      choices and a loop for every choice with experience accumulation. Thus, this
      function is useless if the model requires additional or less choices. For each
      number of choices with and without experience, a new function had to be
      programmed. The following approach uses the same loops over choices with
      experiences, but they are dynamically created by the recursive function
      :func:`_create_core_state_space_per_period`.

    - There are characteristics of the state space which are independent from all other
      state space attributes like types (and almost lagged choices). These attributes
      only duplicate the existing state space and can be taken into account in a later
      stage of the process.

    See also
    --------
    _create_core_state_space
    _create_core_state_space_per_period
    _filter_core_state_space
    _add_initial_experiences_to_core_state_space
    _create_state_space_indexer

    """
    df = _create_core_state_space(options)

    df = _add_lagged_choice_to_core_state_space(df, options)

    df = _filter_core_state_space(df, options)

    df = _add_initial_experiences_to_core_state_space(df, options)

    df = _add_types_to_state_space(df, options["n_types"])

    df = df.sort_values("period").reset_index(drop=True)

    indexer = _create_state_space_indexer(df, options)

    df.lagged_choice = pd.Categorical(df.lagged_choice, categories=options["choices"])

    return df, indexer


def _create_core_state_space(options):
    """Create the core state space.

    The core state space abstracts from initial experiences and uses the maximum range
    between initial experiences and maximum experiences to cover the whole range. The
    combinations of initial experiences are applied later in
    :func:`_add_initial_experiences_to_core_state_space`.

    See also
    --------
    _create_core_state_space_per_period

    """
    minimal_initial_experience = np.array(
        [
            np.min(options["choices"][choice]["start"])
            for choice in options["choices_w_exp"]
        ],
        dtype=np.uint8,
    )
    maximum_exp = np.array(
        [options["choices"][choice]["max"] for choice in options["choices_w_exp"]],
        dtype=np.uint8,
    )

    additional_exp = maximum_exp - minimal_initial_experience

    exp_cols = [f"exp_{choice}" for choice in options["choices_w_exp"]]

    container = []
    for period in np.arange(options["n_periods"], dtype=np.uint8):
        data = _create_core_state_space_per_period(
            period,
            additional_exp,
            options,
            np.zeros(len(options["choices_w_exp"]), dtype=np.uint8),
        )
        df_ = pd.DataFrame.from_records(data, columns=exp_cols)
        df_.insert(0, "period", period)
        container.append(df_)

    df = pd.concat(container, axis="rows", sort=False)

    return df


def _create_core_state_space_per_period(
    period, additional_exp, options, experiences, pos=0
):
    """Create core state space per period.

    First, this function returns a state combined with all possible lagged choices and
    types.

    Secondly, if there exists a choice with experience in ``additional_exp[pos]``, loop
    over all admissible experiences, update the state and pass it to the same function,
    but moving to the next choice which accumulates experience.

    Parameters
    ----------
    period : int
        Number of period.
    additional_exp : numpy.ndarray
        Array with shape (n_choices_w_exp,) containing integers representing the
        additional experience per choice which is admissible. This is the difference
        between the maximum experience and minimum of initial experience per choice.
    experiences : None or numpy.ndarray, default None
        Array with shape (n_choices_w_exp,) which contains current experience of state.
    pos : int, default 0
        Index for current choice with experience. If index is valid for array
        ``experiences``, then loop over all admissible experience levels of this choice.
        Otherwise, ``experiences[pos]`` would lead to an :exc:`IndexError`.

    """
    # Return experiences combined with lagged choices and types.
    yield experiences

    # Check if there is an additional choice left to start another loop.
    if pos < experiences.shape[0]:
        # Upper bound of additional experience is given by the remaining time or the
        # maximum experience which can be accumulated in experience[pos].
        remaining_time = period - experiences.sum()
        max_experience = additional_exp[pos]

        # +1 is necessary so that the remaining time or max_experience is exhausted.
        for i in np.arange(min(remaining_time, max_experience) + 1, dtype=np.uint8):
            # Update experiences and call the same function with the next choice.
            updated_experiences = experiences.copy()
            updated_experiences[pos] += i
            yield from _create_core_state_space_per_period(
                period, additional_exp, options, updated_experiences, pos + 1
            )


def _add_lagged_choice_to_core_state_space(df, options):
    container = []
    for choice in options["choices"]:
        df_ = df.copy()
        df_["lagged_choice"] = choice
        container.append(df_)

    df = pd.concat(container, axis="rows", sort=False)

    return df


def _filter_core_state_space(df, options):
    """Applies filters to the core state space.

    Sometimes, we want to apply filters to a group of choices. Thus, use the following
    shortcuts.

    - ``i`` is replaced with every choice with experience.
    - ``j`` is replaced with every choice without experience.
    - ``k`` is replaced with every choice with a wage.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains the core state space.
    options : dict
        Contains model options and the filters to reduce the core state space.

    """
    for definition in options["core_state_space_filters"]:
        # If "{i}" is in definition, loop over choices with experiences.
        if "{i}" in definition:
            for i in options["choices_w_exp"]:
                df = df.loc[~df.eval(definition.format(i=i))]

        # If "{j}" is in definition, loop over choices without experiences.
        elif "{j}" in definition:
            for j in options["choices_wo_exp"]:
                df = df.loc[~df.eval(definition.format(j=j))]

        # If "{k}" is in definition, loop over choices with wage.
        elif "{k}" in definition:
            for k in options["choices_w_wage"]:
                df = df.loc[~df.eval(definition.format(k=k))]

        else:
            df = df.loc[~df.eval(definition)]

    return df


def _add_initial_experiences_to_core_state_space(df, options):
    """Add initial experiences to core state space.

    As the core state space abstracts from differences in initial experiences, this
    function loops through all combinations from initial experiences and adds them to
    existing experiences. After that, we need to check whether the maximum in
    experiences is still binding.

    """
    # Create combinations of starting values
    initial_experiences_combinations = itertools.product(
        *[options["choices"][choice]["start"] for choice in options["choices_w_exp"]]
    )

    maximum_exp = np.array(
        [options["choices"][choice]["max"] for choice in options["choices_w_exp"]]
    )

    exp_cols = df.filter(like="exp_").columns.tolist()

    container = []

    for initial_exp in initial_experiences_combinations:
        df_ = df.copy()

        # Add initial experiences.
        df_[exp_cols] += initial_exp

        # Check that max_experience is still fulfilled.
        df_ = df_.loc[df_[exp_cols].le(maximum_exp).all(axis="columns")].copy()

        container.append(df_)

    df = pd.concat(container, axis="rows", sort=False).drop_duplicates()

    return df


def _add_types_to_state_space(df, n_types):
    container = []
    for i in range(n_types):
        df_ = df.copy()
        df_["type"] = i
        container.append(df_)

    df = pd.concat(container, axis="rows", sort=False)

    return df


def _create_state_space_indexer(df, options):
    """Create the indexer for the state space.

    The indexer consists of sub indexers for each period. This is much more
    memory-efficient than having a single indexer. For more information see the
    references section.

    References
    ----------
    - https://github.com/OpenSourceEconomics/respy/pull/236
    - https://github.com/OpenSourceEconomics/respy/pull/237

    """
    n_exp_choices = len(options["choices_w_exp"])
    n_nonexp_choices = len(options["choices_wo_exp"])
    choices = options["choices"]

    max_initial_experience = np.array(
        [choices[choice]["start"].max() for choice in options["choices_w_exp"]]
    ).astype(np.uint8)
    max_experience = [choices[choice]["max"] for choice in options["choices_w_exp"]]

    choice_to_code = {choice: i for i, choice in enumerate(options["choices"])}

    indexer = []
    count_states = 0

    for period in range(options["n_periods"]):
        shape = tuple(
            np.minimum(max_initial_experience + period, max_experience) + 1
        ) + (n_exp_choices + n_nonexp_choices, options["n_types"])
        sub_indexer = np.full(shape, -1, dtype=np.int32)

        sub_df = df.loc[df.period.eq(period)]
        n_states = sub_df.shape[0]

        indices = tuple(sub_df[f"exp_{i}"] for i in options["choices_w_exp"]) + (
            sub_df.lagged_choice.replace(choice_to_code),
            sub_df.type,
        )
        sub_indexer[indices] = np.arange(count_states, count_states + n_states)
        indexer.append(sub_indexer)

        count_states += n_states

    return indexer


def _create_reward_components(types, covariates, optim_paras, options):
    """Calculate systematic rewards for each state.

    Wages are only available for some choices, i.e. n_nonpec >= n_wages. We extend the
    array of wages with ones for the difference in dimensions, n_nonpec - n_wages. Ones
    are necessary as it facilitates the aggregation of reward components in
    :func:`calculate_emax_value_functions` and related functions.

    Parameters
    ----------
    types : numpy.ndarray
        Array with shape (n_states,) containing type information.
    covariates : dict
        Dictionary with covariate arrays for wage and nonpec rewards.
    optim_paras : dict
        Contains parameters affected by the optimization.

    """
    wage_labels = [f"wage_{choice}" for choice in options["choices_w_wage"]]
    log_wages = np.column_stack(
        [np.dot(covariates[w], optim_paras[w]) for w in wage_labels]
    )

    n_states = covariates["nonpec_edu"].shape[0]

    nonpec_labels = [f"nonpec_{choice}" for choice in options["choices"]]
    nonpec = np.column_stack(
        [
            np.zeros(n_states)
            if n not in optim_paras
            else np.dot(covariates[n], optim_paras[n])
            for n in nonpec_labels
        ]
    )

    n_wages = len(options["choices_w_wage"])
    type_deviations = optim_paras["type_shift"][types]

    log_wages += type_deviations[:, :n_wages]
    nonpec[:, n_wages:] += type_deviations[:, n_wages:]

    wages = np.clip(np.exp(log_wages), 0.0, HUGE_FLOAT)

    # Extend wages to dimension of non-pecuniary rewards.
    additional_dim = nonpec.shape[1] - log_wages.shape[1]
    wages = np.column_stack((wages, np.ones((wages.shape[0], additional_dim))))

    return wages, nonpec


def _create_choice_covariates(covariates_df, states_df, params, options):
    """Create the covariates for each choice.

    Parameters
    ----------
    covariates_df : pandas.DataFrame
        DataFrame with the basic covariates.
    states_df : pandas.DataFrame
        DataFrame with the state information.
    params : pandas.DataFrame or pandas.Series
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

    for choice in options["choices"]:
        if f"wage_{choice}" in params.index:
            wage_columns = params.loc[f"wage_{choice}"].index
            covariates[f"wage_{choice}"] = all_data[wage_columns].to_numpy()

        if f"nonpec_{choice}" in params.index:
            nonpec_columns = params.loc[f"nonpec_{choice}"].index
            covariates[f"nonpec_{choice}"] = all_data[nonpec_columns].to_numpy()

    for key, val in covariates.items():
        covariates[key] = np.ascontiguousarray(val)

    return covariates


def create_base_covariates(states, covariates_spec):
    """Create set of covariates for each state.

    Parameters
    ----------
    states : pandas.DataFrame
        DataFrame with shape (n_states, n_choices_w_exp + 3) containing period,
        experiences, choice_lagged and type of each state.
    covariates_spec : dict
        Keys represent covariates and values are strings passed to ``df.eval``.

    Returns
    -------
    covariates : pandas.DataFrame
        DataFrame with shape (n_states, n_covariates).

    """
    covariates = states.copy()

    for covariate, definition in covariates_spec.items():
        covariates[covariate] = covariates.eval(definition)

    covariates = covariates.drop(columns=states.columns)

    return covariates


def _create_is_inadmissible_indicator(states, options):
    # Make maximum experience for each choice available as local variable.
    locals_ = {
        f"max_exp_{choice}": options["choices"][choice]["max"]
        for choice in options["choices_w_exp"]
    }

    df = states.copy()

    for column, definition in options["inadmissible_states"].items():
        df[column] = df.eval(definition, local_dict=locals_)

    is_inadmissible = df[options["choices"]].to_numpy()

    return is_inadmissible
