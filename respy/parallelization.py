"""This module contains the code to control parallel execution."""
import copy
import functools

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from joblib import parallel_backend

from respy.shared import create_dense_state_space_columns


def broadcast_arguments(state_space, args, kwargs):
    """Broadcast arguments to dense state space dimensions."""
    args = list(args) if isinstance(args, tuple) else [args]

    for i, arg in enumerate(args):
        if _is_dense_dictionary_argument(arg, state_space.dense):
            pass
        else:
            args[i] = {idx: arg for idx in state_space.dense}
    for kwarg, value in kwargs.items():
        if _is_dense_dictionary_argument(value, state_space.dense):
            pass
        else:
            kwargs[kwarg] = {idx: value for idx in state_space.dense}

    args = [[arg[dense_idx] for arg in args] for dense_idx in state_space.dense]
    kwargs = [
        {kwarg: value[dense_idx] for kwarg, value in kwargs.items()}
        for dense_idx in state_space.dense
    ]

    return args, kwargs


def parallelize_across_dense_dimensions(func):
    """Parallelizes decorated function across dense state space dimensions.

    Parallelization is only possible if the decorated function has no side-effects to
    other dense dimensions. This might be true for different levels. E.g.
    :meth:`respy.state_space.StateSpace.create_choice_rewards` can be directly
    parallelized. :func:`respy.solve.solve_with_backward_induction` cannot be
    directly parallized because the continuation values for one dense dimension will
    become important for others if we implement exogenous processes. Thus, parallelize
    across periods.

    If applied to a function, the decorator recognizes if the model or state space
    contains dense dimensions likes types or observables. Then, it splits the operation
    across dense dimensions by patching the attribute access such that each sub state
    space can only access its attributes.

    The copy is shallow and thus sub state spaces hold references to the original arrays
    which should be changed in-place.

    """

    @functools.wraps(func)
    def wrapper_parallelize_across_dense_dimensions(state_space, *args, **kwargs):
        if state_space.dense:
            state_spaces = []
            for idx in state_space.dense:
                state_space_ = copy.copy(state_space)
                state_space_.get_attribute = functools.partial(
                    state_space_.get_attribute, dense_idx=idx
                )
                state_spaces.append(state_space_)

            args, kwargs = broadcast_arguments(state_space, args, kwargs)

            with parallel_backend("loky"):
                out = Parallel(n_jobs=1, max_nbytes=None)(
                    delayed(func)(st_sp, *args, **kwargs)
                    for st_sp, args, kwargs in zip(state_spaces, args, kwargs)
                )

            out = dict(zip(state_space.dense, out))
        else:
            out = func(state_space, *args, **kwargs)

        return out

    return wrapper_parallelize_across_dense_dimensions


def _is_dense_dictionary_argument(argument, dense_indices):
    return not np.isscalar(argument) and all(idx in argument for idx in dense_indices)


def distribute_and_combine_simulation(func):
    """Distribute the simulation across sub state spaces and combine the outputs."""

    @functools.wraps(func)
    def wrapper_distribute_and_combine_simulation(state_space, df, optim_paras):
        columns = create_dense_state_space_columns(optim_paras)
        if columns:
            groups = df.groupby(columns).groups
            for key in list(groups):
                sub_df = df.loc[groups.pop(key)].copy()
                key = (int(key),) if len(columns) == 1 else tuple(int(i) for i in key)
                groups[key] = sub_df

        else:
            groups = df

        out = func(state_space, groups, optim_paras)

        df = pd.concat(out.values()).sort_index() if isinstance(out, dict) else out

        return df

    return wrapper_distribute_and_combine_simulation
