"""Everything related to the state space of a structural model."""
import itertools

import numba as nb
import numpy as np
import pandas as pd
from numba import types
from numba.typed.typeddict import Dict

from respy._numba import array_to_tuple
from respy.parallelization import parallelize_across_dense_dimensions
from respy.shared import compute_covariates
from respy.shared import convert_dictionary_keys_to_dense_indices
from respy.shared import create_base_draws
from respy.shared import create_core_state_space_columns
from respy.shared import create_dense_state_space_columns
from respy.shared import downcast_to_smallest_dtype
from respy.shared import return_core_dense_key
from respy.shared import subset_to_period


def create_state_space_class(optim_paras, options):
    """Take all preperations for creating a sp object and create it."""
    core = _create_core_and_indexer(optim_paras, options)
    dense_grid = _create_dense_state_space_grid(optim_paras)

    # Downcast after calculations or be aware of silent integer overflows.
    core = compute_covariates(core, options["covariates_core"])
    core = core.apply(downcast_to_smallest_dtype)
    dense = _create_dense_state_space_covariates(dense_grid, optim_paras, options)

    # Create choice_period_core
    core_period_choice = _create_core_period_choice(core, optim_paras, options)

    # I think here we can get more elegant! Or is this the only way?
    core_index_to_complex = dict(enumerate(core_period_choice))
    core_index_to_indices = {
        i: core_period_choice[core_index_to_complex[i]] for i in core_index_to_complex
    }

    # Create sp indexer
    indexer = _create_indexer(core, core_index_to_indices, optim_paras)

    dense_period_choice = _create_dense_period_choice(
        core, dense, core_index_to_indices, core_index_to_complex, optim_paras, options
    )

    state_space = StateSpaceClass(
        core,
        indexer,
        dense,
        dense_period_choice,
        core_index_to_complex,
        core_index_to_indices,
        optim_paras,
        options,
    )

    return state_space


class StateSpaceClass:
    """Explain how things work once finally decided upon."""

    def __init__(
        self,
        core,
        indexer,
        dense,
        dense_period_cores,
        core_index_to_complex,
        core_index_to_indices,
        optim_paras,
        options,
    ):
        """Insert a description of variables once this is final."""
        self.core = core
        self.indexer = indexer
        self.dense_period_cores = dense_period_cores
        self.dense = dense
        self.core_index_to_complex = core_index_to_complex
        self.core_index_to_indices = core_index_to_indices
        self.optim_paras = optim_paras
        self.options = options
        self.set_convenience_objects()
        self.child_indices = self.collect_child_indices()
        self.base_draws_sol = self.create_draws()
        self.create_expected_value_func()

    def set_convenience_objects(self):
        """Create a number of conveneience objects."""
        self.dense_to_dense_idx = (
            {} if not self.dense else {i: k for i, k in enumerate(self.dense)}
        )

        self.index_to_complex = {i: k for i, k in enumerate(self.dense_period_cores)}

        self.index_to_core_index = {
            i: self.dense_period_cores[self.index_to_complex[i]]
            for i in self.index_to_complex.keys()
        }

        self.index_to_choice_set = {
            i: self.index_to_complex[i][1] for i in self.index_to_complex
        }

        self.index_to_indices = {
            i: np.array(self.core_index_to_indices[self.index_to_core_index[i]])
            for i in self.index_to_complex
        }

        self.core_to_index = Dict.empty(
            key_type=types.UniTuple(types.int64, 2), value_type=types.int64,
        )

        for i in self.index_to_complex:
            self.core_to_index[
                return_core_dense_key(
                    self.index_to_core_index[i], *self.index_to_complex[i][2:]
                )
            ] = i

        self.complex_to_index = {k: i for i, k in self.index_to_complex.items()}

        if self.dense is False:
            self.dense_covariates_to_index = {}
            self.index_to_dense_covariates = {i: {} for i in self.index_to_complex}

        else:
            n_dense = len(create_dense_state_space_columns(self.optim_paras))
            self.dense_covariates_to_index = Dict.empty(
                key_type=types.UniTuple(types.int64, n_dense), value_type=types.int64,
            )
            for i, k in enumerate(self.dense):
                self.dense_covariates_to_index[k] = i

            self.index_to_dense_covariates = {
                i: list(self.dense.values())[self.index_to_complex[i][2]]
                for i in self.index_to_complex
            }

    def create_expected_value_func(self):
        """Create a contianer for expected value functions."""
        self.expected_value_functions = Dict.empty(
            key_type=types.int64, value_type=types.float64[:]
        )
        for index, indices in self.index_to_indices.items():
            self.expected_value_functions[index] = np.zeros(len(indices))

    def get_continuation_values(self, period):
        """Wrap get continuation values."""
        if period == self.options["n_periods"] - 1:
            shapes = self.get_attribute_from_period("base_draws_sol", period)
            states = self.get_attribute_from_period("index_to_indices", period)
            continuation_values = {
                key: np.zeros((states[key].shape[0], shapes[key].shape[1]))
                for key in shapes
            }
        else:
            child_indices = self.get_attribute_from_period("child_indices", period)
            expected_value_functions = self.get_attribute_from_period(
                "expected_value_functions", period + 1
            )
            subset_expected_value_functions = Dict.empty(
                key_type=types.int64, value_type=types.float64[:]
            )
            for key, value in expected_value_functions.items():
                subset_expected_value_functions[key] = value

            continuation_values = _get_continuation_values(
                self.get_attribute_from_period("index_to_indices", period),
                self.get_attribute_from_period("index_to_complex", period),
                child_indices,
                self.core_to_index,
                bypass={"expected_value_functions": subset_expected_value_functions},
            )
        return continuation_values

    def collect_child_indices(self):
        """Wrap get child indices."""
        limited_index_to_indices = {
            k: v
            for k, v in self.index_to_indices.items()
            if self.index_to_complex[k][0] < self.options["n_periods"] - 1
        }

        limited_index_to_choice_set = {
            k: v
            for k, v in self.index_to_choice_set.items()
            if self.index_to_complex[k][0] < self.options["n_periods"] - 1
        }

        child_indices = _collect_child_indices(
            self.core,
            limited_index_to_indices,
            limited_index_to_choice_set,
            self.indexer,
            self.optim_paras,
            self.options,
        )

        return child_indices

    def create_draws(self):
        """Get draws."""
        n_choices_in_sets = np.unique(
            [sum(i) for i in self.index_to_choice_set.values()]
        ).tolist()
        shocks_sets = []

        for n_choices in n_choices_in_sets:
            draws = create_base_draws(
                (self.options["n_periods"], self.options["solution_draws"], n_choices),
                next(self.options["solution_seed_startup"]),
                self.options["monte_carlo_sequence"],
            )
            shocks_sets.append(draws)
        draws = {}
        for dense_idx, complex_ix in self.index_to_complex.items():
            period = complex_ix[0]
            n_choices = sum(complex_ix[1])
            idx = n_choices_in_sets.index(n_choices)
            draws[dense_idx] = shocks_sets[idx][period]

        return draws

    def get_attribute(self, attr):
        """Get an attribute of the state space."""
        return getattr(self, attr)

    def get_attribute_from_period(self, attribute, period):
        """Get an attribute of the state space sliced to a given period.
        Parameters
        ----------
        attribute : str
            Attribute name, e.g. ``"states"`` to retrieve ``self.states``.
        period : int
            Attribute is retrieved from this period.
        """
        attr = self.get_attribute(attribute)
        out = subset_to_period(attr, self.index_to_complex, period)
        return out

    def get_attribute_from_key(self, attribute, key):
        """Get an attribute of the state space sliced to a given period.
        Parameters
        ----------
        attribute : str
            Attribute name, e.g. ``"states"`` to retrieve ``self.states``.
        period : int
            Attribute is retrieved from this period.
        """
        attr = self.get_attribute(attribute)
        out = attr[key]
        return out

    def set_attribute(self, attribute, value):
        """Set attributes."""
        setattr(self, attribute, value)

    def set_attribute_from_keys(self, attribute, value):
        """Set attributes by keys."""
        for key in value.keys():
            self.get_attribute_from_key(attribute, key)[:] = value[key]


def _create_core_and_indexer(optim_paras, options):
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
    _create_core_state_space_indexer

    """
    core = _create_core_state_space(optim_paras)

    core = _add_lagged_choice_to_core_state_space(core, optim_paras)

    core = _filter_core_state_space(core, options)

    core = _add_initial_experiences_to_core_state_space(core, optim_paras)

    core = core.sort_values("period").reset_index(drop=True)

    return core


def _create_core_state_space(optim_paras):
    """Create the core state space.

    The core state space abstracts from initial experiences and uses the maximum range
    between initial experiences and maximum experiences to cover the whole range. The
    combinations of initial experiences are applied later in
    :func:`_add_initial_experiences_to_core_state_space`.

    See also
    --------
    _create_core_state_space_per_period

    """

    choices_w_exp = list(optim_paras["choices_w_exp"])
    minimal_initial_experience = np.array(
        [min(optim_paras["choices"][choice]["start"]) for choice in choices_w_exp],
        dtype=np.uint8,
    )
    maximum_exp = np.array(
        [optim_paras["choices"][choice]["max"] for choice in choices_w_exp],
        dtype=np.uint8,
    )

    additional_exp = maximum_exp - minimal_initial_experience

    exp_cols = [f"exp_{choice}" for choice in choices_w_exp]

    container = []
    for period in np.arange(optim_paras["n_periods"], dtype=np.uint8):
        data = _create_core_state_space_per_period(
            period,
            additional_exp,
            optim_paras,
            np.zeros(len(choices_w_exp), dtype=np.uint8),
        )
        df_ = pd.DataFrame.from_records(data, columns=exp_cols)
        df_.insert(0, "period", period)
        container.append(df_)

    df = pd.concat(container, axis="rows", sort=False)

    return df


def _create_core_state_space_per_period(
    period, additional_exp, optim_paras, experiences, pos=0
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
                period, additional_exp, optim_paras, updated_experiences, pos + 1
            )


def _add_lagged_choice_to_core_state_space(df, optim_paras):
    container = []
    for lag in range(1, optim_paras["n_lagged_choices"] + 1):
        for choice_code in range(len(optim_paras["choices"])):
            df_ = df.copy()
            df_[f"lagged_choice_{lag}"] = choice_code
            container.append(df_)

    df = pd.concat(container, axis="rows", sort=False) if container else df

    return df


def _filter_core_state_space(df, options):
    """Apply filters to the core state space.

    Sometimes, we want to apply filters to a group of choices. Thus, use the following
    shortcuts.

    - ``i`` is replaced with every choice with experience.
    - ``j`` is replaced with every choice without experience.
    - ``k`` is replaced with every choice with a wage.

    Parameters
    ----------
    df : pandas.DataFrame
    options : dict

    """
    for definition in options["core_state_space_filters"]:
        df = df.loc[~df.eval(definition)]

    return df


def _add_initial_experiences_to_core_state_space(df, optim_paras):
    """Add initial experiences to core state space.

    As the core state space abstracts from differences in initial experiences, this
    function loops through all combinations from initial experiences and adds them to
    existing experiences. After that, we need to check whether the maximum in
    experiences is still binding.

    """
    choices = optim_paras["choices"]
    # Create combinations of starting values
    initial_experiences_combinations = itertools.product(
        *[choices[choice]["start"] for choice in optim_paras["choices_w_exp"]]
    )

    maximum_exp = np.array(
        [choices[choice]["max"] for choice in optim_paras["choices_w_exp"]]
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


def _create_dense_state_space_grid(optim_paras):
    levels_of_observables = [range(len(i)) for i in optim_paras["observables"].values()]
    types = [range(optim_paras["n_types"])] if optim_paras["n_types"] >= 2 else []

    dense_state_space_grid = list(itertools.product(*levels_of_observables, *types))
    if dense_state_space_grid == [()]:
        dense_state_space_grid = False

    return dense_state_space_grid


def _create_dense_state_space_covariates(dense_grid, optim_paras, options):
    if dense_grid:
        columns = create_dense_state_space_columns(optim_paras)

        df = pd.DataFrame(data=dense_grid, columns=columns).set_index(
            columns, drop=False
        )

        covariates = compute_covariates(df, options["covariates_dense"])
        covariates = covariates.apply(downcast_to_smallest_dtype)
        covariates = covariates.to_dict(orient="index")
        covariates = convert_dictionary_keys_to_dense_indices(covariates)

    else:
        covariates = False

    return covariates


def create_is_inadmissible(df, optim_paras, options):
    for choice in optim_paras["choices"]:
        df[f"_{choice}"] = False
        for formula in options["inadmissible_states"][choice]:
            try:
                df[f"_{choice}"] |= df.eval(formula)
            except pd.core.computation.ops.UndefinedVariableError:
                pass

        # df[f"_{choice}"] = df[f"_{choice}"].astype(np.bool_)

    return df


def _split_core_state_space(core, optim_paras):
    """
    This function splits the sp according to period and choice set
    """
    periodic_cores = {idx: sub for idx, sub in core.groupby("period")}
    periodic_choice_cores = {
        (period, choice_set): sub.index
        for period in periodic_cores
        for choice_set, sub in periodic_cores[period].groupby(
            list(optim_paras["choices"])
        )
    }

    return periodic_choice_cores


def _create_indexer(core, core_index_to_indices, optim_paras):
    core_columns = ["period"] + create_core_state_space_columns(optim_paras)
    n_core_state_variables = len(core_columns)

    indexer = Dict.empty(
        key_type=types.UniTuple(types.int64, n_core_state_variables),
        value_type=types.UniTuple(types.int64, 2),
    )

    for core_idx, indices in core_index_to_indices.items():
        states = core.loc[indices, core_columns].to_numpy()
        for i, state in enumerate(states):
            indexer[tuple(state)] = (core_idx, i)
    return indexer


def _create_core_period_choice(core, optim_paras, options):
    """

    """
    choices = [f"_{choice}" for choice in optim_paras["choices"]]
    df = core.copy()
    df = create_is_inadmissible(df, optim_paras, options)
    df[choices] = ~df[choices]
    core_period_choice = {
        (idx[0], idx[1:]): indices
        for idx, indices in df.groupby(["period"] + choices).groups.items()
    }
    return core_period_choice


def _create_dense_period_choice(
    core, dense, core_index_to_indices, core_index_to_complex, optim_paras, options
):
    """

    """
    if dense is False:
        return {k: i for i, k in core_index_to_complex.items()}
    else:
        choices = [f"_{choice}" for choice in optim_paras["choices"]]
        dense_period_choice = {}
        for core_idx, indices in core_index_to_indices.items():
            for dense_idx, (_, dense_vec) in enumerate(dense.items()):
                df = core.loc[indices].copy().assign(**dense_vec)
                df = compute_covariates(df, options["covariates_all"])
                df = create_is_inadmissible(df, optim_paras, options)
                df[choices] = ~df[choices]
                grouper = df.groupby(choices).groups
                assert len(grouper) == 1, (
                    "Choice restrictions cannot interact between core"
                    " and dense information such that heterogenous choice"
                    " sets within a period are created. Use penalties in the "
                    "utility functions for that."
                )
                period_choice = {
                    (core_index_to_complex[core_idx][0], idx, dense_idx): core_idx
                    for idx, indices in grouper.items()
                }

                dense_period_choice = {**dense_period_choice, **period_choice}

    return dense_period_choice


@nb.njit
def _insert_indices_of_child_states(
    states, indexer, choice_set, n_choices, n_choices_w_exp, n_lagged_choices,
):
    """Collect indices of child states for each parent state."""
    indices = np.full((states.shape[0], n_choices, 2), -1, dtype=np.int64)

    for i in range(states.shape[0]):
        j = 0
        for choice, is_available in enumerate(choice_set):
            if is_available:
                child = states[i].copy()

                # Adjust period.
                child[0] += 1

                # Increment experience if it is a choice with experience
                # accumulation.
                if choice < n_choices_w_exp:
                    child[choice + 1] += 1

                # Change lagged choice by shifting all existing lagged choices by
                # one period and inserting the current choice in first position.
                if n_lagged_choices:
                    child[
                        n_choices_w_exp + 2 : n_choices_w_exp + n_lagged_choices + 1
                    ] = child[n_choices_w_exp + 1 : n_choices_w_exp + n_lagged_choices]
                    child[n_choices_w_exp + 1] = choice

                # Get the position of the continuation value.
                idx_future = indexer[array_to_tuple(indexer, child)]
                indices[i, j] = idx_future

                j += 1

    return indices


@parallelize_across_dense_dimensions
@nb.njit(parallel=True)
def _get_continuation_values(
    core_indices,
    dense_complex_index,
    child_indices,
    core_index_and_dense_vector_to_dense_index,
    expected_value_functions,
):

    if len(dense_complex_index) == 3:
        period, choice_set, dense_idx = dense_complex_index
    elif len(dense_complex_index) == 2:
        period, choice_set = dense_complex_index
        dense_idx = 0

    # Sum over UniTuple.
    n_choices = 0
    for x in choice_set:
        if x:
            n_choices += 1

    n_states = core_indices.shape[0]

    continuation_values = np.zeros((len(core_indices), n_choices))
    for i in nb.prange(n_states):
        for j in nb.prange(n_choices):
            core_idx, row_idx = child_indices[i, j]
            idx = (core_idx, dense_idx)
            dense_choice = core_index_and_dense_vector_to_dense_index[idx]

            continuation_values[i, j] = expected_value_functions[dense_choice][row_idx]

    return continuation_values


@parallelize_across_dense_dimensions
def _collect_child_indices(
    core, core_indices, choice_set, indexer, optim_paras, options
):
    n_choices = sum(choice_set)

    core_columns = create_core_state_space_columns(optim_paras)
    states = core.loc[core_indices, ["period"] + core_columns].to_numpy()

    indices = _insert_indices_of_child_states(
        states,
        indexer,
        choice_set,
        n_choices,
        len(optim_paras["choices_w_exp"]),
        optim_paras["n_lagged_choices"],
    )

    return indices
