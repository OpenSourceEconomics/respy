"""Everything related to the state space of a structural model."""
import itertools

import numba as nb
import numpy as np
import pandas as pd
from numba.typed import Dict

from respy._numba import sum_over_numba_boolean_unituple
from respy.exogenous_processes import create_transit_choice_set
from respy.exogenous_processes import create_transition_objects
from respy.exogenous_processes import weight_continuation_values
from respy.parallelization import parallelize_across_dense_dimensions
from respy.shared import apply_law_of_motion_for_core
from respy.shared import compute_covariates
from respy.shared import convert_dictionary_keys_to_dense_indices
from respy.shared import create_base_draws
from respy.shared import create_core_state_space_columns
from respy.shared import create_dense_state_space_columns
from respy.shared import downcast_to_smallest_dtype
from respy.shared import dump_objects
from respy.shared import get_choice_set_from_complex
from respy.shared import load_objects
from respy.shared import map_states_to_core_key_and_core_index
from respy.shared import prepare_cache_directory
from respy.shared import return_core_dense_key


def create_state_space_class(optim_paras, options):
    """Create the state space of the model."""
    prepare_cache_directory(options)
    core = _create_core_state_space(optim_paras, options)
    dense_grid = _create_dense_state_space_grid(optim_paras)

    # Downcast after calculations or be aware of silent integer overflows.
    core = compute_covariates(core, options["covariates_core"])
    core = core.apply(downcast_to_smallest_dtype)
    dense = _create_dense_state_space_covariates(dense_grid, optim_paras, options)

    core_period_choice = _create_core_period_choice(core, optim_paras, options)

    core_key_to_complex = dict(enumerate(core_period_choice))
    core_key_to_core_indices = {
        i: core_period_choice[complex_] for i, complex_ in core_key_to_complex.items()
    }

    indexer = _create_indexer(core, core_key_to_core_indices, optim_paras)

    dense_period_choice = _create_dense_period_choice(
        core, dense, core_key_to_core_indices, core_key_to_complex, optim_paras, options
    )

    state_space = StateSpace(
        core,
        indexer,
        dense,
        dense_period_choice,
        core_key_to_complex,
        core_key_to_core_indices,
        optim_paras,
        options,
    )

    return state_space


class StateSpace:
    """The state space of a structural model.

    Attributes
    ----------
    core : dict of pandas.DataFrame
        The core state space is a :class:`pandas.DataFrame` that contains all states of
        core dimensions. A core dimension is a dimension whose value is uniquely
        determined by past choices and time. Core dimensions include choices,
        experiences, lagged choices and periods.
    dense_key_to_core_indices : Dict[int, Array[int]]
        A mapping from dense keys to ``.loc`` locations in the ``core``.

    """

    def __init__(
        self,
        core,
        indexer,
        dense,
        dense_period_cores,
        core_key_to_complex,
        core_key_to_core_indices,
        optim_paras,
        options,
    ):
        """Initialize the state space.

        See :ref:`state space location indices <state_space_location_indices>`
        for an explanation of the dict objects.

        Parameters
        ----------
        core : pandas.DataFrame
            DataFrame containing one core state per row.
        indexer : numba.typed.Dict
            Maps states (rows of core) into tuples containing core key and
            core index. i : state -> (core_key, core_index)
        dense : dict
            Maps dense states into dense covariates.
        dense_period_cores : dict
            Maps period, choice_set and dense_index into core_key.
            d : (core_key,choice_set,dense_index) -> core_key
        core_key_to_complex : dict
            Maps period and choice_set into core_key
        core_key_to_core_indices : dict
            Maps core_keys into core_indices.

        """
        self.core = core
        self.indexer = indexer
        self.dense_period_cores = dense_period_cores
        self.dense = dense
        self.core_key_to_complex = core_key_to_complex
        self.core_key_to_core_indices = core_key_to_core_indices
        self.optim_paras = optim_paras
        self.options = options
        self.n_periods = options["n_periods"]
        self._create_conversion_dictionaries()
        self.base_draws_sol = self.create_draws(options)
        self.create_arrays_for_expected_value_functions()

        if len(self.optim_paras["exogenous_processes"]) > 0:
            self.create_objects_for_exogenous_processes()
        self.child_indices = self.collect_child_indices()

    def _create_conversion_dictionaries(self):
        """Create mappings between state space location indices and properties.

        See :ref:`state space location indices <state_space_location_indices>`.

        """
        self.dense_key_to_complex = {
            i: k for i, k in enumerate(self.dense_period_cores)
        }

        self.dense_key_to_core_key = {
            i: self.dense_period_cores[self.dense_key_to_complex[i]]
            for i in self.dense_key_to_complex
        }

        self.dense_key_to_choice_set = {
            i: self.dense_key_to_complex[i][1] for i in self.dense_key_to_complex
        }

        self.dense_key_to_core_indices = {
            i: np.array(self.core_key_to_core_indices[self.dense_key_to_core_key[i]])
            for i in self.dense_key_to_complex
        }

        self.core_key_and_dense_index_to_dense_key = Dict.empty(
            key_type=nb.types.UniTuple(nb.types.int64, 2), value_type=nb.types.int64,
        )

        for i in self.dense_key_to_complex:
            self.core_key_and_dense_index_to_dense_key[
                return_core_dense_key(
                    self.dense_key_to_core_key[i], *self.dense_key_to_complex[i][2:],
                )
            ] = i

        if self.dense is False:
            self.dense_covariates_to_dense_index = {}
            self.dense_key_to_dense_covariates = {
                i: {} for i in self.dense_key_to_complex
            }

        else:
            n_dense = len(create_dense_state_space_columns(self.optim_paras))
            self.dense_covariates_to_dense_index = Dict.empty(
                key_type=nb.types.UniTuple(nb.types.int64, n_dense),
                value_type=nb.types.int64,
            )
            for i, k in enumerate(self.dense):
                self.dense_covariates_to_dense_index[k] = i

            self.dense_key_to_dense_covariates = {
                i: list(self.dense.keys())[self.dense_key_to_complex[i][2]]
                for i in self.dense_key_to_complex
            }

    def create_arrays_for_expected_value_functions(self):
        """Create a container for expected value functions."""
        self.expected_value_functions = Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.float64[:]
        )
        for index, indices in self.dense_key_to_core_indices.items():
            self.expected_value_functions[index] = np.zeros(len(indices))

    def create_objects_for_exogenous_processes(self):
        """Insert Docstring Here."""
        # Include switch arg

        exogenous_processes = self.optim_paras["exogenous_processes"]
        n_exog = len(exogenous_processes)

        # How does the accounting work here again? Would that actually work?
        levels_of_processes = [range(len(i)) for i in exogenous_processes.values()]
        self.exogenous_grid = list(itertools.product(*levels_of_processes))

        self.dense_key_to_transit_keys = create_transition_objects(
            self.dense_key_to_dense_covariates,
            self.dense_key_to_core_key,
            self.exogenous_grid,
            n_exog,
            bypass={
                "dense_covariates_to_dense_index": self.dense_covariates_to_dense_index,
                "core_key_and_dense_index_to_dense_key": self.core_key_and_dense_index_to_dense_key,  # noqa: E501
            },
        )
        self.transit_key_to_choice_set = create_transit_choice_set(
            self.dense_key_to_transit_keys, self.dense_key_to_complex
        )

    def get_continuation_values(self, period):
        """Get continuation values.

        The function takes the expected value functions from the previous periods and
        then uses the indices of child states to put these expected value functions in
        the correct format. If period is equal to self.n_periods - 1 the function
        returns arrays of zeros since we are in terminal states. Otherwise we retrieve
        expected value functions for next period and call
        :func:`_get_continuation_values` to assign continuation values to all choices
        within a period. (The object `subset_expected_value_functions` is required
        because we need a Numba typed dict but the function
        :meth:`StateSpace.get_attribute_from_period` just returns a normal dict)

        Returns
        -------
        continuation_values : numba.typed.Dict
            The continuation values for each dense key in a :class:`numpy.ndarray`.

        See also
        --------
        _get_continuation_values
            A more theoretical explanation can be found here: See :ref:`get continuation
            values <get_continuation_values>`.

        """
        if period == self.n_periods - 1:
            shapes = self.get_attribute_from_period("base_draws_sol", period)
            states = self.get_attribute_from_period("dense_key_to_core_indices", period)
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
                key_type=nb.types.int64, value_type=nb.types.float64[:]
            )
            for key, value in expected_value_functions.items():
                subset_expected_value_functions[key] = value

            transit_choice_sets = (
                "transit_key_to_choice_set"
                if hasattr(self, "transit_key_to_choice_set")
                else "dense_key_to_choice_set"
            )

            continuation_values = _get_continuation_values(
                self.get_attribute_from_period("dense_key_to_complex", period),
                self.get_attribute_from_period(transit_choice_sets, period),
                self.get_attribute_from_period("dense_key_to_core_indices", period),
                child_indices,
                self.core_key_and_dense_index_to_dense_key,
                bypass={"expected_value_functions": subset_expected_value_functions},
            )

            if len(self.optim_paras["exogenous_processes"]) > 0:
                continuation_values = weight_continuation_values(
                    self.get_attribute_from_period("dense_key_to_complex", period),
                    self.options,
                    bypass={
                        "continuation_values": continuation_values,
                        "transit_key_to_choice_set": self.get_attribute_from_period(
                            transit_choice_sets, period
                        ),
                    },
                )

        return continuation_values

    def collect_child_indices(self):
        """Collect for each state the indices of its child states.

        To collect continuation values, one needs to find the child state. This function
        searches for all states except for the last period their possible successors by
        taking every possible combination defined by the law of motion.

        See also
        --------
        _collect_child_indices
            A more theoretical explanation can be found here: See :ref:`collect child
            indices <collect_child_indices>`.

        """
        if self.n_periods == 1:
            child_indices = None

        else:
            dense_key_to_complex_except_last_period = {
                k: v
                for k, v in self.dense_key_to_complex.items()
                if v[0] < self.n_periods - 1
            }

            transit_choice_sets = (
                "transit_key_to_choice_set"
                if hasattr(self, "transit_key_to_choice_set")
                else "dense_key_to_choice_set"
            )

            dense_key_to_choice_set_except_last_period = {
                k: getattr(self, transit_choice_sets)[k]
                for k in dense_key_to_complex_except_last_period
            }

            child_indices = _collect_child_indices(
                dense_key_to_complex_except_last_period,
                dense_key_to_choice_set_except_last_period,
                self.indexer,
                self.optim_paras,
                self.options,
            )

        return child_indices

    def create_draws(self, options):
        """Get draws."""
        all_choice_sets = map(
            get_choice_set_from_complex, self.dense_key_to_complex.values()
        )
        n_choices_in_sets = list(set(map(sum, all_choice_sets)))
        shocks_sets = []

        for n_choices in n_choices_in_sets:
            draws = create_base_draws(
                (options["n_periods"], options["solution_draws"], n_choices),
                next(options["solution_seed_startup"]),
                options["monte_carlo_sequence"],
            )
            shocks_sets.append(draws)
        draws = {}
        for dense_idx, complex_ix in self.dense_key_to_complex.items():
            period = complex_ix[0]
            n_choices = sum(complex_ix[1])
            idx = n_choices_in_sets.index(n_choices)
            draws[dense_idx] = shocks_sets[idx][period]

        return draws

    def get_dense_keys_from_period(self, period):
        """Get dense indices from one period."""
        return [
            dense_index
            for dense_index, complex_ in self.dense_key_to_complex.items()
            if complex_[0] == period
        ]

    def get_attribute_from_period(self, attribute, period):
        """Get an attribute of the state space sliced to a given period.

        Parameters
        ----------
        attribute : str
            Attribute name, e.g. ``"states"`` to retrieve ``self.states``.
        period : int
            Attribute is retrieved from this period.

        """
        dense_indices_in_period = self.get_dense_keys_from_period(period)
        return {
            dense_index: attr
            for dense_index, attr in getattr(self, attribute).items()
            if dense_index in dense_indices_in_period
        }

    def set_attribute_from_keys(self, attribute, value):
        """Set attributes by keys.

        This function allows to modify the period part of a certain state space object.
        It allows to set values for all dense period choice cores within one period.
        During the model solution this method in period :math:`t + 1` communicates with
        get continuation values in period :math:`t`.

        Note that the values are changed in-place.

        Parameters
        ----------
        attribute : str
            The name of the state space attribute which is changed in-place.
        value : numpy.ndarray
            The value to which the Numpy array is set.

        """
        for key in value:
            getattr(self, attribute)[key][:] = value[key]


def _create_core_state_space(optim_paras, options):
    """Create the core state space.

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
    _create_core_from_choice_experiences
    _create_core_state_space_per_period
    _filter_core_state_space
    _add_initial_experiences_to_core_state_space
    _create_indexer

    """
    core = _create_core_from_choice_experiences(optim_paras)

    core = _add_lagged_choice_to_core_state_space(core, optim_paras)

    core = _filter_core_state_space(core, options)

    core = _add_initial_experiences_to_core_state_space(core, optim_paras)

    core = core.sort_values("period").reset_index(drop=True)

    return core


def _create_core_from_choice_experiences(optim_paras):
    """Create the core state space from choice experiences.

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
    """Create a grid of dense variables.

    The function loops through all potential realizations of each dense dimension and
    returns a list of all possible joint realizations of dense variables.

    Parameters
    ----------
    optim_paras : dict
        Contains parsed model parameters.

    Returns
    -------
    dense_state_space_grid : list
        Contains all dense states as tuples.

    """
    levels_of_observables = [range(len(i)) for i in optim_paras["observables"].values()]
    types = [range(optim_paras["n_types"])] if optim_paras["n_types"] >= 2 else []

    dense_state_space_grid = list(itertools.product(*levels_of_observables, *types))
    if dense_state_space_grid == [()]:
        dense_state_space_grid = False

    return dense_state_space_grid


def _create_dense_state_space_covariates(dense_grid, optim_paras, options):
    """Obtain covariates for all dense states."""
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
    """Compute is_inadmissible for passed states."""
    df = df.copy()

    for choice in optim_paras["choices"]:
        df[f"_{choice}"] = False
        for formula in options["negative_choice_set"][choice]:
            try:
                df[f"_{choice}"] |= df.eval(formula)
            except pd.core.computation.ops.UndefinedVariableError:
                pass

    return df


def _create_indexer(core, core_key_to_core_indices, optim_paras):
    """Create indexer of core state space.

    Returns
    -------
    indexer :  numba.typed.Dict
        Maps a row of the core state space into its position within the
        period_choice_cores. c: core_state -> (core_key,core_index)

    """
    core_columns = ["period"] + create_core_state_space_columns(optim_paras)
    n_core_state_variables = len(core_columns)

    indexer = Dict.empty(
        key_type=nb.types.UniTuple(nb.types.int64, n_core_state_variables),
        value_type=nb.types.UniTuple(nb.types.int64, 2),
    )

    for core_idx, indices in core_key_to_core_indices.items():
        states = core.loc[indices, core_columns].to_numpy()
        for i, state in enumerate(states):
            indexer[tuple(state)] = (core_idx, i)
    return indexer


def _create_core_period_choice(core, optim_paras, options):
    """Create the core separated into period-choice cores.

    Returns
    -------
    core_period_choice : dict
        c: (period, choice_set) -> core_indices

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
    core, dense, core_key_to_core_indices, core_key_to_complex, optim_paras, options
):
    """Create dense period choice parts of the state space.

    We loop over all dense combinations and calculate choice restrictions for each
    particular dense state space. The information allows us to compile a dict that maps
    a combination of period, choice_set and dense_index into core_key!

    Note that we do not allow for choice restrictions that interact between core and
    dense covariates. In order to do so we would have to rewrite this function and
    return explicit state space position instead of core indices!

    Returns
    -------
    dense_period_choice : dict
        d: (period, choice_set, dense_index) -> core_key

    """
    if not dense:
        for key, complex_ in core_key_to_complex.items():
            dump_objects(
                core.loc[core_key_to_core_indices[key]], "states", complex_, options
            )
        dense_period_choice = {k: i for i, k in core_key_to_complex.items()}
    else:
        choices = [f"_{choice}" for choice in optim_paras["choices"]]
        dense_period_choice = {}
        for dense_idx, (_, dense_vec) in enumerate(dense.items()):
            states = core.copy().assign(**dense_vec)
            states = compute_covariates(states, options["covariates_all"])

            states = create_is_inadmissible(states, optim_paras, options)
            for core_idx, indices in core_key_to_core_indices.items():
                df = states.copy().loc[indices].assign(**dense_vec)
                df[choices] = ~df[choices]
                grouper = df.groupby(choices).groups
                if not len(grouper) == 1:
                    raise ValueError(
                        "Choice restrictions cannot interact between core and dense "
                        "information such that heterogeneous choice sets within a "
                        "period are created. Use penalties in the utility functions "
                        "for that."
                    )
                period_choice = {
                    (core_key_to_complex[core_idx][0], idx, dense_idx): core_idx
                    for idx, indices in grouper.items()
                }

                dense_period_choice = {**dense_period_choice, **period_choice}
                idx = list(grouper.keys())[0]
                dump_objects(
                    df,
                    "states",
                    (core_key_to_complex[core_idx][0], idx, dense_idx),
                    options,
                )

    return dense_period_choice


@parallelize_across_dense_dimensions
def _get_continuation_values(
    dense_complex_index,
    choice_set,
    core_indices,
    child_indices,
    core_index_and_dense_vector_to_dense_index,
    expected_value_functions,
):
    """Get continuation values from child states.

    The continuation values are the discounted expected value functions from child
    states. This method allows to retrieve continuation values that were obtained in the
    model solution. In particular the function assigns continuation values to state
    choice combinations by using the child indices created in
    :func:`_collect_child_indices`.

    Returns
    -------
    continuation_values : numpy.ndarray
        Array with shape ``(n_states, n_choices)``. Maps core_key and choice into
        continuation value.

    """
    if len(dense_complex_index) == 3:
        period, _, dense_idx = dense_complex_index
    elif len(dense_complex_index) == 2:
        period, _ = dense_complex_index
        dense_idx = 0

    n_choices = sum_over_numba_boolean_unituple(choice_set)

    n_states = core_indices.shape[0]

    continuation_values = np.zeros((len(core_indices), n_choices))
    for i in range(n_states):
        for j in range(n_choices):
            core_idx, row_idx = child_indices[i, j]
            idx = (core_idx, dense_idx)
            dense_choice = core_index_and_dense_vector_to_dense_index[idx]

            continuation_values[i, j] = expected_value_functions[dense_choice][row_idx]

    return continuation_values


@parallelize_across_dense_dimensions
def _collect_child_indices(complex_, choice_set, indexer, optim_paras, options):
    """Collect child indices for states.

    The function takes the states of one dense key, applies the law of motion for each
    available choice and maps the resulting states to core keys and core indices.

    Parameters
    ----------
    complex_ : tuple
        See :ref:`complex`.
    choice_set : tuple
        Tuple representing admissible choices
    indexer : numba.typed.Dict
        A dictionary with core states as keys and the core key and core index as values.
    optim_paras : dict
        Contains model parameters.
    options : dict
        Contains model options.

    Returns
    -------
    indices : numpy.ndarray
        Array with shape ``(n_states, n_choices * 2)``. Represents the mapping
        (core_index, choice) -> (dense_key, core_index).

    """
    core_columns = create_core_state_space_columns(optim_paras)
    states = load_objects("states", complex_, options)

    n_choices = sum(choice_set)
    indices = np.full((states.shape[0], n_choices, 2), -1, dtype=np.int64)

    indices_valid_choices = [i for i, is_valid in enumerate(choice_set) if is_valid]
    for i, choice in enumerate(indices_valid_choices):
        states_ = states.copy(deep=True)

        states_["choice"] = choice
        states_ = apply_law_of_motion_for_core(states_, optim_paras)

        states_ = states_[["period"] + core_columns]

        indices[:, i, 0], indices[:, i, 1] = map_states_to_core_key_and_core_index(
            states_.to_numpy(), indexer
        )

    return indices
