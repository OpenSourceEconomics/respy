"""Everything related to the state space of a structural model."""
import itertools
import warnings

import numba as nb
import numpy as np
import pandas as pd

from respy._numba import array_to_tuple
from respy.config import INADMISSIBILITY_PENALTY
from respy.config import INDEXER_DTYPE
from respy.config import INDEXER_INVALID_INDEX
from respy.shared import compute_covariates
from respy.shared import convert_dictionary_keys_to_dense_indices
from respy.shared import create_base_draws
from respy.shared import create_core_state_space_columns
from respy.shared import create_dense_state_space_columns
from respy.shared import downcast_to_smallest_dtype


def create_state_space_class(optim_paras, options):
    """Create the state space of the model."""
    core, indexer = _create_core_and_indexer(optim_paras, options)
    dense_grid = _create_dense_state_space_grid(optim_paras)

    # Downcast after calculations or be aware of silent integer overflows.
    core = compute_covariates(core, options["covariates_core"])
    core = core.apply(downcast_to_smallest_dtype)
    dense = _create_dense_state_space_covariates(dense_grid, optim_paras, options)

    base_draws_sol = create_base_draws(
        (options["n_periods"], options["solution_draws"], len(optim_paras["choices"])),
        next(options["solution_seed_startup"]),
        options["monte_carlo_sequence"],
    )

    if dense:
        state_space = _MultiDimStateSpace(
            core, indexer, base_draws_sol, optim_paras, options, dense
        )
    else:
        state_space = _SingleDimStateSpace(
            core, indexer, base_draws_sol, optim_paras, options
        )

    return state_space


def create_sp(options, optim_paras):
    """
    Define a function that calls all the prior methods to create something
    that looks like a sp!
    """
    # We create the full core first! (Do we have to create covariates here?)
    core, indexer = _create_core_and_indexer(optim_paras, options)

    # We apply all inadmissable states
    core = compute_covariates(core, options["covariates_core"])
    core = core.apply(downcast_to_smallest_dtype)
    core = _create_choice_sets(core, optim_paras, options, "core")

    # Create dense sp!
    dense_grid = _create_dense_state_space_grid(optim_paras)
    dense_cov = _create_dense_state_space_covariates(dense_grid, optim_paras, options)

    # Create base draws sol
    base_draws_sol = create_base_draws(
        (options["n_periods"], options["solution_draws"], len(optim_paras["choices"])),
        next(options["solution_seed_startup"]),
        options["monte_carlo_sequence"],
    )

    #TODO: We have to change code here such as to allow for no dense vars.
    #      We need to take new strcuture of dense variables into account!
    state_space = _MultiDimStateSpace(
        core,
        indexer,
        base_draws_sol,
        optim_paras,
        options,
        dense_cov
        )

    return state_space


class _BaseStateSpace:
    """The base class of a state space.
    The base class includes some methods which should be available to both state spaces
    and are shared between multiple sub state spaces.
    """

    def _create_indices_of_child_states(self, optim_paras):
        """For each parent state get the indices of child states.
        During the backward induction, the ``expected_value_functions`` in the future
        period serve as the ``continuation_values`` of the current period. As the
        indices for child states never change, these indices can be precomputed and
        added to the state_space.
        Actually, the indices of the child states do not have to cover the last period,
        but it makes the code prettier and reduces the need to expand the indices in the
        estimation.
        """
        n_choices = len(optim_paras["choices"])
        n_choices_w_exp = len(optim_paras["choices_w_exp"])
        n_periods = optim_paras["n_periods"]
        n_states = self.core.shape[0]
        core_columns = create_core_state_space_columns(optim_paras)

        indices = np.full(
            (n_states, n_choices), INDEXER_INVALID_INDEX, dtype=INDEXER_DTYPE
        )

        # Skip the last period which does not have child states.
        for period in reversed(range(n_periods - 1)):
            states_in_period = self.core.query("period == @period")[
                core_columns
            ].to_numpy(dtype=np.int8)

            indices = _insert_indices_of_child_states(
                indices,
                states_in_period,
                self.indexer[period],
                self.indexer[period + 1],
                len(self.optim_paras["choices"]),
                n_choices_w_exp,
                optim_paras["n_lagged_choices"],
            )

        return indices


class _SingleDimStateSpace(_BaseStateSpace):
    """
    This is the state space that handles one period of data!
    """
    def __init__(
        self,
        core,
        indexer,
        base_draws_sol,
        optim_paras,
        options,
        dense_dim = None,
        dense_covariates = None,
        indices_of_child_states = None,
        ):
        self.core = core
        self.indexer = indexer
        self.base_draws = base_draws_sol
        self.optim_paras = optim_paras
        self.options = options
        self.dense_dim = dense_dim
        self.dense_covariates = dense_covariates if dense_covariates is not None else {}
        self.mixed_covariates = options["covariates_mixed"]
        self.indices_of_child_states = (
            super()._create_indices_of_child_states(optim_paras)
            if indices_of_child_states is None
            else indices_of_child_states
        )
        self.expected_value_functions = np.empty(self.core.shape[0])


    def get_continuation_values(self, period):
        n_periods = len(self.indexer)
        choice_cores = {
            key[1]: self.period_choice_cores[key] for key in self.period_choice_cores.keys()
            if key[0] == period
        }
        continuation_values = dict()
        for choice_set in choice_cores.keys():
            if period == n_periods - 1:
                last_slice = choice_cores[choice_set]
                n_states_last_period = len(last_slice)
                continuation_values_choice = np.zeros((n_states_last_period, choice_set.count(True)))
            else:
                child_indices = self.get_attribute("indices_of_child_states")[choice_cores[choice_set], choice_set]
                mask = child_indices != INDEXER_INVALID_INDEX
                valid_indices = np.where(mask, child_indices, 0)
                continuation_values_choice = np.where(
                    mask, self.get_attribute("expected_value_functions")[valid_indices], 0
                )
            continuation_values[choice_set] = continuation_values_choice
        return continuation_values


    def get_attribute(self, attr):
        """Get an attribute of the state space."""
        return getattr(self, attr)


    def get_attribute_from_period_choice(self, attribute, period, choice_set, privacy="public"):
        attr = self.get_attribute(attribute)
        if privacy == "public":
            slice_ = self.period_choice_cores[(period, choice_set)]
            out = attr[slice_]
        elif privacy == "privat":
            out = attr[(period,choice_set)]
        return out


    def set_attribute(self, attribute, value):
        self.get_attribute(attribute)[:] = value


    def set_attribute_from_period_choice_set(self, attribute, value, period, choice_set, privacy):
        self.get_attribute_from_period_choice(attribute, period, choice_set, privacy)[:] = value


    @property
    def states(self):
        states = self.core.copy().assign(**self.dense_covariates)
        states = compute_covariates(states, self.mixed_covariates)
        return states

    @property
    def period_choice_cores(self):
        if self.dense_dim:
            print(self.dense_covariates)
            dense_df = pd.DataFrame(self.dense_covariates,index=[0])
            dense_df = _create_choice_sets(dense_df, self.optim_paras, self.options, "dense")
            dense_choice_set = {key:dense_df[key] for key in self.optim_paras["choices"]}
            # Set all to False in core!
            states = self.states
            for choice in dense_choice_set.keys():
                print(dense_choice_set[choice])
                if dense_choice_set[choice].loc[0] == True:
                    states[choice] = True

        states = _create_choice_sets(states, self.optim_paras, self.options, "mixed")
        # Dirty Fix to reverse False and True
        for choice in self.optim_paras["choices"]:
            states[choice] = ~ states[choice]
        period_choice_cores = _split_core_state_space(states,self.optim_paras)

        return period_choice_cores

    @property
    def base_draws_sol(self):
        return _reshape_base_draws(self.base_draws, self.period_choice_cores, self.options)



class _MultiDimStateSpace(_BaseStateSpace):
    """The state space of a discrete choice dynamic programming model.
    This class wraps the whole state space of the model.
    In this case we take the whole seperated choice space and create individual objects for each
    dense option. Operations are still seperated and thus the old logic still largely applies.
    Down the road we might want to redcue the number of input variables since this looks a bit confusing now
    """

    def __init__(self,
                 core,
                 indexer,
                 base_draws,
                 optim_paras,
                 options,
                 dense
                 ):
        self.core = core
        self.indexer = indexer
        self.base_draws = base_draws
        self.optim_paras = optim_paras
        self.options = options
        self.dense = dense
        self.indices_of_child_states = super()._create_indices_of_child_states(
            optim_paras
        )
        self.sub_state_spaces = {
            dense_dim: _SingleDimStateSpace(
                self.core.copy(),
                self.indexer,
                self.base_draws.copy(),
                optim_paras,
                options,
                dense_dim,
                dense_covariates,
                self.indices_of_child_states,
            )
            for dense_dim, dense_covariates in dense.items()}

    def get_attribute(self, attribute):
        return {
            key: sss.get_attribute(attribute)
            for key, sss in self.sub_state_spaces.items()
        }

    def get_attribute_from_period_choice_set(self, attribute, period, choice_set, privacy):
        return {
            key: sss.get_attribute_from_period(attribute, period, choice_set, privacy)
            for key, sss in self.sub_state_spaces.items()
        }

    def get_continuation_values(self, period, indices=None):
        return {
            key: sss.get_continuation_values(period, indices)
            for key, sss in self.sub_state_spaces.items()
        }

    def set_attribute(self, attribute, value):
        for key, sss in self.sub_state_spaces.items():
            sss.set_attribute(attribute, value[key])

    def set_attribute_from_period_choice_set(self, attribute, period, choice_set, value, privacy):
        for key, sss in self.sub_state_spaces.items():
            sss.set_attribute_from_choice_set_private(
                attribute, value[key], period, choice_set, privacy)

    @property
    def states(self):
        return {key: sss.states for key, sss in self.sub_state_spaces.items()}
    @property
    def period_choice_cores(self):
        return {key: sss.period_choice_cores for key, sss in self.sub_state_spaces.items()}
    @property
    def base_draws_sol(self):
        return {key: sss.base_draws_sol for key, sss in self.sub_state_spaces.items()}

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

    indexer = _create_core_state_space_indexer(core, optim_paras)

    return core, indexer


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


def _create_core_state_space_indexer(df, optim_paras):
    """Create the indexer for the state space.

    The indexer consists of sub indexers for each period. This is much more
    memory-efficient than having a single indexer. For more information see the
    references section.

    References
    ----------
    - https://github.com/OpenSourceEconomics/respy/pull/236
    - https://github.com/OpenSourceEconomics/respy/pull/237

    """
    n_choices = len(optim_paras["choices"])
    choices = optim_paras["choices"]

    max_initial_experience = np.array(
        [max(choices[choice]["start"]) for choice in optim_paras["choices_w_exp"]]
    ).astype(np.uint8)
    max_experience = [choices[choice]["max"] for choice in optim_paras["choices_w_exp"]]

    indexer = []
    count_states = 0

    for period in range(optim_paras["n_periods"]):
        shape = (
            tuple(np.minimum(max_initial_experience + period, max_experience) + 1)
            + (n_choices,) * optim_paras["n_lagged_choices"]
        )
        sub_indexer = np.full(shape, INDEXER_INVALID_INDEX, dtype=INDEXER_DTYPE)

        sub_df = df.query("period == @period")
        n_states = sub_df.shape[0]

        indices = tuple(
            sub_df[f"exp_{i}"] for i in optim_paras["choices_w_exp"]
        ) + tuple(
            sub_df[f"lagged_choice_{i}"]
            for i in range(1, optim_paras["n_lagged_choices"] + 1)
        )

        sub_indexer[indices] = np.arange(count_states, count_states + n_states)
        indexer.append(sub_indexer)

        count_states += n_states

    return indexer


@nb.njit
def _insert_indices_of_child_states(
    indices,
    states,
    indexer_current,
    indexer_future,
    n_choices_tot,
    n_choices_w_exp,
    n_lagged_choices,
):
    """Collect indices of child states for each parent state."""
    n_choices = n_choices_tot

    for i in range(states.shape[0]):

        idx_current = indexer_current[array_to_tuple(indexer_current, states[i])]

        for choice in range(n_choices):

            # Cut off the period which is not necessary for the indexer.
            child = states[i].copy()

            # Increment experience if it is a choice with experience
            # accumulation.
            if choice < n_choices_w_exp:
                child[choice] += 1

            # Change lagged choice by shifting all existing lagged choices by
            # one period and inserting the current choice in first position.
            if n_lagged_choices:
                child[
                    n_choices_w_exp + 1 : n_choices_w_exp + n_lagged_choices
                ] = child[n_choices_w_exp : n_choices_w_exp + n_lagged_choices - 1]
                child[n_choices_w_exp] = choice

            # Get the position of the continuation value.
            idx_future = indexer_future[array_to_tuple(indexer_future, child)]
            indices[idx_current, choice] = idx_future

    return indices


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


def _create_choice_sets(df, optim_paras, options, category):
    """
    Todo: Check to which extent we can use the same procedure for
          core and dense!
    """
    df = df.copy()

    # Apply user-defined constraints

    for choice in optim_paras["choices"]:
        if choice in df.columns:
            continue
        else:
            df[choice] = False

    for choice in options[f"inadmissible_choices_{category}"].keys():
        for formula in options[f"inadmissible_choices_{category}"][choice]:
            try:
                df[choice] |= df.eval(formula[0])
            except pd.core.computation.ops.UndefinedVariableError:
                pass
    return df


def _split_core_state_space(core, optim_paras):
    """
    This function splits the sp according to period and choice set
    """
    periodic_cores = {idx: sub for idx, sub in core.groupby("period")}
    periodic_choice_cores = {(period, choice_set): sub
                             for period in periodic_cores
                             for choice_set, sub
                             in periodic_cores[period].groupby(list(optim_paras["choices"]))
                             }

    return periodic_choice_cores


def _get_subspace_of_choice(choice_set):
    """
    This is WIP!
    """
    out = []
    choice_set = np.array(choice_set)
    check = np.where(np.invert(choice_set))

    for x in itertools.product([True, False], repeat=len(choice_set)):
        if all(np.all(y) == False for y in np.array(x)[check]):
            out.append(x)
        else:
            pass
    return out


def _reshape_base_draws(draws, period_choice_cores, options):
    """
    Distribute draws across period choice core subsets.
    Highly preliminary for illustrative purposes!
    """
    size_dict = {key: len(period_choice_cores[key]) for key in period_choice_cores.keys()}
    n_states_draws = draws.shape[1]
    out = dict()
    for period in range(options["n_periods"]):
        prop_dict = {key: size_dict[key] for key in size_dict.keys() if key[0] == period}
        n_states_core = np.array(list(prop_dict.values())).sum()
        pos = 0
        for key in prop_dict.keys():
            num_states = int(np.ceil(prop_dict[key] * n_states_draws / n_states_core))
            if pos + num_states > n_states_draws:
                num_states = n_states_draws - pos
            out[key] = draws[period, slice(pos, pos + num_states), key[1]]
            pos = pos + num_states

    return out
