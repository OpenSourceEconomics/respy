"""This module contains the code to control parallel execution."""
import functools

import joblib
import numpy as np
import pandas as pd

from respy.shared import create_dense_state_space_columns


def parallelize_across_dense_dimensions(func=None, *, n_jobs=1):
    """Parallelizes decorated function across dense state space dimensions.

    Parallelization is only possible if the decorated function has no side-effects to
    other dense dimensions. This might be true for different levels. E.g.
    :meth:`respy.solve._create_choice_rewards` can be directly
    parallelized. :func:`respy.solve.solve_with_backward_induction` cannot be
    directly parallelized because the continuation values for one dense dimension will
    become important for others if we implement exogenous processes. Thus, parallelize
    across periods.

    If applied to a function, the decorator recognizes if the model or state space
    contains dense dimensions likes types or observables. Then, it splits the operation
    across dense dimensions by patching the attribute access such that each sub state
    space can only access its attributes.

    The decorator can be applied to functions without trailing parentheses. At the same
    time, the `*` prohibits to use the decorator with positional arguments.

    """

    def decorator_parallelize_across_dense_dimensions(func):
        @functools.wraps(func)
        def wrapper_parallelize_across_dense_dimensions(*args, **kwargs):
            dense_indices = _infer_dense_indices_from_arguments(args, kwargs)
            if dense_indices:
                args_, kwargs_ = _broadcast_arguments(args, kwargs, dense_indices)

                out = joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)(
                    joblib.delayed(func)(*args_[idx], **kwargs_[idx])
                    for idx in dense_indices
                )

                # Re-order multiple return values from list of tuples to tuple of lists
                # to tuple of dictionaries to set as state space attributes.
                if isinstance(out[0], tuple):
                    n_returns = len(out[0])
                    tuple_of_lists = tuple(
                        [single_out[i] for single_out in out] for i in range(n_returns)
                    )
                    out = tuple(
                        dict(zip(dense_indices, list_)) for list_ in tuple_of_lists
                    )
                else:
                    out = dict(zip(dense_indices, out))
            else:
                out = func(*args, **kwargs)

            return out

        return wrapper_parallelize_across_dense_dimensions

    # Ensures that the decorator can be used without parentheses.
    if callable(func):
        return decorator_parallelize_across_dense_dimensions(func)
    else:
        return decorator_parallelize_across_dense_dimensions


def _infer_dense_indices_from_arguments(args, kwargs):
    """Infer the dense indices from the arguments."""
    dense_indices = []
    for arg in args:
        if _is_dictionary_with_tuple_keys(arg):
            dense_indices.extend(arg.keys())
            break
    if not dense_indices:
        for kwarg in kwargs.values():
            if _is_dictionary_with_tuple_keys(kwarg):
                dense_indices.extend(kwarg.keys())
                break

    return dense_indices


def _is_dictionary_with_tuple_keys(candidate):
    """Infer whether the argument is a dictionary with tuple keys."""
    return isinstance(candidate, dict) and all(
        isinstance(key, tuple) for key in candidate
    )


def _broadcast_arguments(args, kwargs, dense_indices):
    """Broadcast arguments to dense state space dimensions."""
    args = list(args) if isinstance(args, tuple) else [args]

    # Broadcast arguments which are not captured in a dictionary with dense state space
    # dimension as keys.
    for i, arg in enumerate(args):
        if _is_dense_dictionary_argument(arg, dense_indices):
            pass
        else:
            args[i] = {idx: arg for idx in dense_indices}
    for kwarg, value in kwargs.items():
        if _is_dense_dictionary_argument(value, dense_indices):
            pass
        else:
            kwargs[kwarg] = {idx: value for idx in dense_indices}

    # Re-order arguments for zipping.
    args = {dense_idx: [arg[dense_idx] for arg in args] for dense_idx in dense_indices}
    kwargs = {
        dense_idx: {kwarg: value[dense_idx] for kwarg, value in kwargs.items()}
        for dense_idx in dense_indices
    }

    return args, kwargs


def _is_dense_dictionary_argument(argument, dense_indices):
    """Check whether all keys of the dictionary argument are also dense indices.

    We cannot check whether all dense indices are in the argument because `splitted_df`
    in :func:`distribute_and_combine_simulation` may not cover all dense combinations.

    """
    return isinstance(argument, dict) and all(idx in dense_indices for idx in argument)


def distribute_and_combine_simulation(func):
    """Distribute the simulation across sub state spaces and combine."""

    @functools.wraps(func)
    def wrapper_distribute_and_combine_simulation(df, *args, optim_paras, **kwargs):
        dense_columns = create_dense_state_space_columns(optim_paras)
        splitted_df = _split_dataframe(df, dense_columns) if dense_columns else df

        out = func(splitted_df, *args, optim_paras, **kwargs)

        df = pd.concat(out.values()).sort_index() if isinstance(out, dict) else out

        return df

    return wrapper_distribute_and_combine_simulation


def distribute_and_combine_likelihood(func):
    """Distribute the likelihood calculation across sub state spaces and combine."""

    @functools.wraps(func)
    def wrapper_distribute_and_combine_likelihood(
        df, base_draws_est, *args, optim_paras, options
    ):
        dense_columns = create_dense_state_space_columns(optim_paras)
        # Duplicate the DataFrame for each type.
        if dense_columns:
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
                type_specific_dense = _split_dataframe(df_, dense_columns)
                splitted_df = {**splitted_df, **type_specific_dense}
            splitted_shocks = _split_shocks(
                base_draws_est, splitted_df, indices, optim_paras
            )
        else:
            splitted_df = df
            splitted_shocks = base_draws_est

        out = func(splitted_df, splitted_shocks, *args, optim_paras, options)

        out = pd.concat(out.values()).sort_index() if isinstance(out, dict) else out

        return out

    return wrapper_distribute_and_combine_likelihood


def _split_dataframe(df, group_columns):
    groups = df.groupby(group_columns).groups
    for key in list(groups):
        sub_df = df.loc[groups.pop(key)].copy()
        key = (int(key),) if len(group_columns) == 1 else tuple(int(i) for i in key)
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
