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


class _BaseStateSpace:
    """The base class of a state space.

    The base class includes some methods which should be available to both state spaces
    and are shared between multiple sub state spaces.

    """

    def _create_slices_by_core_periods(self):
        """Create slices to index all attributes in a given period.

        It is important that the returned objects are not fancy indices. Fancy indexing
        results in copies of array which decrease performance and raise memory usage.

        """
        period = self.core.period
        indices = np.where(period - period.shift(1).fillna(-1) == 1)[0]
        indices = np.append(indices, self.core.shape[0])

        slices = [slice(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]

        return slices

    def _create_is_inadmissible(self, optim_paras, options):
        core = self.core.copy()
        # Apply the maximum experience as a default constraint only if its not the last
        # period. Otherwise, it is unconstrained.
        for choice in optim_paras["choices_w_exp"]:
            max_exp = optim_paras["choices"][choice]["max"]
            formula = (
                f"exp_{choice} == {max_exp}"
                if max_exp != optim_paras["n_periods"] - 1
                else "False"
            )
            core[choice] = core.eval(formula)

        # Apply no constraint for choices without experience.
        for choice in optim_paras["choices_wo_exp"]:
            core[choice] = core.eval("False")

        # Apply user-defined constraints
        for choice in optim_paras["choices"]:
            for formula in options["inadmissible_states"].get(choice, []):
                core[choice] |= core.eval(formula)

        is_inadmissible = core[optim_paras["choices"]].to_numpy(dtype=np.bool)

        if np.any(is_inadmissible) and optim_paras["inadmissibility_penalty"] is None:
            warnings.warn(
                "'params' does not include a penalty to the utility if an individual "
                "makes an invalid choice. A default penalty of "
                f"{INADMISSIBILITY_PENALTY} is set. Although unproblematic for the full"
                " solution, choose a milder penalty for the approximate solution of the"
                " model or the penalty will break/dominate the linear model for the "
                "interpolation. Set the penalty in your csv with:\n\n"
                "inadmissibility_penalty,inadmissibility_penalty,<negative-number>\n",
                category=UserWarning,
            )

        return is_inadmissible

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
                self.is_inadmissible,
                n_choices_w_exp,
                optim_paras["n_lagged_choices"],
            )

        return indices


class _SingleDimStateSpace(_BaseStateSpace):
    """The state space of a discrete choice dynamic programming model.

    Parameters
    ----------
    core : pandas.DataFrame
        DataFrame containing the core state space.
    indexer : numpy.ndarray
        Multidimensional array containing indices of states in valid positions.
    optim_paras : dict
        Dictionary containing model parameters.
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
    expected_value_functions : numpy.ndarray
        Array with shape (n_states, 1) containing the expected maximum of
        choice-specific value functions.

    """

    def __init__(
        self,
        core,
        indexer,
        base_draws_sol,
        optim_paras,
        options,
        dense_dim=None,
        dense_covariates=None,
        is_inadmissible=None,
        indices_of_child_states=None,
        slices_by_periods=None,
    ):
        self.dense_dim = dense_dim
        self.core = core
        self.indexer = indexer
        self.dense_covariates = dense_covariates if dense_covariates is not None else {}
        self.mixed_covariates = options["covariates_mixed"]
        self.base_draws_sol = base_draws_sol
        self.slices_by_periods = (
            super()._create_slices_by_core_periods()
            if slices_by_periods is None
            else slices_by_periods
        )
        self._initialize_attributes(optim_paras)
        self.is_inadmissible = (
            self._create_is_inadmissible(optim_paras, options)
            if is_inadmissible is None
            else is_inadmissible
        )
        self.indices_of_child_states = (
            super()._create_indices_of_child_states(optim_paras)
            if indices_of_child_states is None
            else indices_of_child_states
        )

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
        slice_ = self.slices_by_periods[period]
        out = attr[slice_]

        return out

    def get_continuation_values(self, period=None, indices=None):
        """Return the continuation values for a given period or states.

        If the last period is selected, return a matrix of zeros. In any other period,
        use the precomputed `indices_of_child_states` to select continuation values from
        `expected_value_functions`.

        You can also indices to collect continuation values across periods.

        Indices may contain :data:`respy.config.INDEXER_INVALID_INDEX` as an identifier
        for invalid states. In this case, indexing leads to an `IndexError`. Replace the
        continuation values with zeros for such indices.

        Parameters
        ----------
        period : int or None
            Return the continuation values for period `period` which are the expected
            value functions from period `period + 1`.
        indices : numpy.ndarray or None
            Indices of states for which to return the continuation values. These states
            can cover multiple periods.

        """
        n_periods = len(self.indexer)

        if period == n_periods - 1:
            last_slice = self.slices_by_periods[-1]
            n_states_last_period = len(range(last_slice.start, last_slice.stop))
            n_choices = self.get_attribute("is_inadmissible").shape[1]
            continuation_values = np.zeros((n_states_last_period, n_choices))

        else:
            if indices is not None:
                child_indices = self.get_attribute("indices_of_child_states")[indices]
            elif period is not None and 0 <= period <= n_periods - 2:
                child_indices = self.get_attribute_from_period(
                    "indices_of_child_states", period
                )
            else:
                raise NotImplementedError

            mask = child_indices != INDEXER_INVALID_INDEX
            valid_indices = np.where(mask, child_indices, 0)
            continuation_values = np.where(
                mask, self.get_attribute("expected_value_functions")[valid_indices], 0
            )

        return continuation_values

    def set_attribute(self, attribute, value):
        self.get_attribute(attribute)[:] = value

    def set_attribute_from_period(self, attribute, value, period):
        self.get_attribute_from_period(attribute, period)[:] = value

    @property
    def states(self):
        states = self.core.copy().assign(**self.dense_covariates)
        states = compute_covariates(states, self.mixed_covariates)
        return states

    def _initialize_attributes(self, optim_paras):
        """Initialize attributes to use references later."""
        n_states = self.core.shape[0]
        n_choices = len(optim_paras["choices"])
        n_choices_w_wage = len(optim_paras["choices_w_wage"])
        n_choices_wo_wage = n_choices - n_choices_w_wage

        for name, array in (
            ("expected_value_functions", np.empty(n_states)),
            (
                "wages",
                np.column_stack(
                    (
                        np.empty((n_states, n_choices_w_wage)),
                        np.ones((n_states, n_choices_wo_wage)),
                    )
                ),
            ),
            ("nonpecs", np.zeros((n_states, n_choices))),
        ):
            setattr(self, name, array)


class _MultiDimStateSpace(_BaseStateSpace):
    """The state space of a discrete choice dynamic programming model.

    This class wraps the whole state space of the model.

    """

    def __init__(self, core, indexer, base_draws_sol, optim_paras, options, dense):
        self.base_draws_sol = base_draws_sol
        self.core = core
        self.indexer = indexer
        self.is_inadmissible = super()._create_is_inadmissible(optim_paras, options)
        self.indices_of_child_states = super()._create_indices_of_child_states(
            optim_paras
        )
        self.slices_by_periods = super()._create_slices_by_core_periods()
        self.sub_state_spaces = {
            dense_dim: _SingleDimStateSpace(
                self.core,
                self.indexer,
                self.base_draws_sol,
                optim_paras,
                options,
                dense_dim,
                dense_covariates,
                self.is_inadmissible,
                self.indices_of_child_states,
                self.slices_by_periods,
            )
            for dense_dim, dense_covariates in dense.items()
        }

    def get_attribute(self, attribute):
        return {
            key: sss.get_attribute(attribute)
            for key, sss in self.sub_state_spaces.items()
        }

    def get_attribute_from_period(self, attribute, period):
        return {
            key: sss.get_attribute_from_period(attribute, period)
            for key, sss in self.sub_state_spaces.items()
        }

    def get_continuation_values(self, period=None, indices=None):
        return {
            key: sss.get_continuation_values(period, indices)
            for key, sss in self.sub_state_spaces.items()
        }

    def set_attribute(self, attribute, value):
        for key, sss in self.sub_state_spaces.items():
            sss.set_attribute(attribute, value[key])

    def set_attribute_from_period(self, attribute, value, period):
        for key, sss in self.sub_state_spaces.items():
            sss.set_attribute_from_period(attribute, value[key], period)

    @property
    def states(self):
        return {key: sss.states for key, sss in self.sub_state_spaces.items()}


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
    is_inadmissible,
    n_choices_w_exp,
    n_lagged_choices,
):
    """Collect indices of child states for each parent state."""
    n_choices = is_inadmissible.shape[1]

    for i in range(states.shape[0]):

        idx_current = indexer_current[array_to_tuple(indexer_current, states[i])]

        for choice in range(n_choices):
            # Check if the state in the future is admissible.
            if is_inadmissible[idx_current, choice]:
                continue
            else:
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

        # Convert scalar keys to tuples.
        for key in covariates.copy():
            if np.isscalar(key):
                covariates[(key,)] = covariates.pop(key)

    else:
        covariates = False

    return covariates
