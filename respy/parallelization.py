"""This module contains the code to control parallel execution."""
import copy
import functools

import numpy as np
import pandas as pd

from respy.shared import create_dense_state_space_columns


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
            args_, kwargs_, dense_indices = _broadcast_arguments(
                state_space, args, kwargs
            )
            sub_state_spaces = _create_dict_of_patched_sub_state_spaces(
                state_space, dense_indices
            )

            out = {}
            for idx in dense_indices:
                out[idx] = func(sub_state_spaces[idx], *args_[idx], **kwargs_[idx])
        else:
            out = func(state_space, *args, **kwargs)

        return out

    return wrapper_parallelize_across_dense_dimensions


def _create_dict_of_patched_sub_state_spaces(state_space, dense_indices):
    """Create a dictionary of state spaces with patched attribute access.

    To parallelize functions across the dense state space, one could pass the dense
    index to functions which then perform the correct lookup. But, it also messes up
    code by introducing the index as a new argument to each function and lookups for
    state space attributes get longer because they need the index.

    Instead, we create shallow copies of the state space class which does only create a
    copy of the class, but not attributes or NumPy arrays attached to it. Then, we patch
    the attribute access such that each sub state space only accesses the rewards and
    expected value functions which belong to its dense state space index.

    """
    sub_state_spaces = {}
    for idx in dense_indices:
        sub_state_space = copy.copy(state_space)
        sub_state_space.get_attribute = functools.partial(
            sub_state_space.get_attribute, dense_idx=idx
        )
        sub_state_spaces[idx] = sub_state_space

    return sub_state_spaces


def _broadcast_arguments(state_space, args, kwargs):
    """Broadcast arguments to dense state space dimensions."""
    args = list(args) if isinstance(args, tuple) else [args]

    # First, infer the necessary dense indices by inspecting `args` and `kwargs`.
    # Functions in the solution operate over all indices, but the simulation might only
    # cover a subset.
    dense_indices = set()
    for arg in args:
        if _is_dense_dictionary_argument(arg, state_space.dense):
            dense_indices |= set(arg)
    for value in kwargs.values():
        if _is_dense_dictionary_argument(value, state_space.dense):
            dense_indices |= set(value)
    dense_indices = dense_indices if dense_indices else state_space.dense.keys()

    # Secondly, broadcast arguments which are not captured in a dictionary with dense
    # state space dimension as keys.
    for i, arg in enumerate(args):
        if _is_dense_dictionary_argument(arg, state_space.dense):
            pass
        else:
            args[i] = {idx: arg for idx in dense_indices}
    for kwarg, value in kwargs.items():
        if _is_dense_dictionary_argument(value, state_space.dense):
            pass
        else:
            kwargs[kwarg] = {idx: value for idx in dense_indices}

    # Thirdly, re-order arguments for zipping.
    args = {dense_idx: [arg[dense_idx] for arg in args] for dense_idx in dense_indices}
    kwargs = {
        dense_idx: {kwarg: value[dense_idx] for kwarg, value in kwargs.items()}
        for dense_idx in dense_indices
    }

    return args, kwargs, dense_indices


def distribute_and_combine_simulation(func):
    """Distribute the simulation across sub state spaces and combine."""

    @functools.wraps(func)
    def wrapper_distribute_and_combine_simulation(state_space, df, optim_paras):
        splitted_df = _split_dataframe(df, optim_paras) if state_space.dense else df

        out = func(state_space, splitted_df, optim_paras)

        df = pd.concat(out.values()).sort_index() if isinstance(out, dict) else out

        return df

    return wrapper_distribute_and_combine_simulation


def distribute_and_combine_likelihood(func):
    """Distribute the likelihood calculation across sub state spaces and combine."""

    @functools.wraps(func)
    def wrapper_distribute_and_combine_likelihood(
        state_space, df, base_draws_est, optim_paras, options
    ):
        # Duplicate the DataFrame for each type.
        if state_space.dense:
            n_obs = df.shape[0]
            n_types = optim_paras["n_types"]
            # Number each state to split the shocks later.
            df["__id"] = np.arange(n_obs)
            # Each row of indices corresponds to a state whereas the columns refer to
            # different types.
            indices = np.arange(n_obs * n_types).reshape(n_obs, n_types)
            splitted_df = {}
            for i in range(optim_paras["n_types"]):
                df_ = df.copy().assign(type=i)
                type_specific_dense = _split_dataframe(df_, optim_paras)
                splitted_df = {**splitted_df, **type_specific_dense}
            splitted_shocks = _split_shocks(
                base_draws_est, splitted_df, indices, optim_paras
            )
        else:
            splitted_df = df
            splitted_shocks = base_draws_est

        out = func(state_space, splitted_df, splitted_shocks, optim_paras, options)

        out = pd.concat(out.values()).sort_index() if isinstance(out, dict) else out

        return out

    return wrapper_distribute_and_combine_likelihood


def _is_dense_dictionary_argument(argument, dense_indices):
    """Check whether all keys of the dictionary argument are also dense indices.

    We cannot check whether all dense indices are in the argument because `splitted_df`
    in :func:`distribute_and_combine_simulation` may not cover all dense combinations.

    """
    return isinstance(argument, dict) and all(idx in dense_indices for idx in argument)


def _split_dataframe(df, optim_paras):
    columns = create_dense_state_space_columns(optim_paras)
    groups = df.groupby(columns).groups
    for key in list(groups):
        sub_df = df.loc[groups.pop(key)].copy()
        key = (int(key),) if len(columns) == 1 else tuple(int(i) for i in key)
        groups[key] = sub_df

    return groups


def _split_shocks(base_draws_est, splitted_df, indices, optim_paras):
    splitted_shocks = {}
    for dense_idx, sub_df in splitted_df.items():
        type_ = dense_idx[-1] if optim_paras["n_types"] >= 2 else 0
        sub_indices = sub_df.pop("__id").to_numpy()
        shock_indices_for_group = indices[sub_indices][:, type_].reshape(-1)
        splitted_shocks[dense_idx] = base_draws_est[shock_indices_for_group]

    return splitted_shocks
