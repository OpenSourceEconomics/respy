"""This module contains the code to control parallel execution."""
import functools

import joblib
import numpy as np
import pandas as pd

from respy.shared import convert_dictionary_keys_to_dense_indices
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

                out = joblib.Parallel(n_jobs=n_jobs)(
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


def split_and_combine_df(func=None, *, remove_type=False):
    """Split the data across sub state spaces and combine.

    This function groups the data according to the dense variables. `remove_type` is
    used to prevent grouping by types which might not be possible.

    """

    def decorator_split_and_combine_df(func):
        @functools.wraps(func)
        def wrapper_distribute_and_combine_df(df, *args, optim_paras, **kwargs):
            dense_columns = create_dense_state_space_columns(optim_paras)
            if remove_type:
                dense_columns.remove("type")

            splitted_df = _split_dataframe(df, dense_columns) if dense_columns else df
            out = func(splitted_df, *args, optim_paras, **kwargs)
            df = pd.concat(out.values()).sort_index() if isinstance(out, dict) else out

            return df

        return wrapper_distribute_and_combine_df

    # Ensures that the decorator can be used without parentheses.
    if callable(func):
        return decorator_split_and_combine_df(func)
    else:
        return decorator_split_and_combine_df


def split_and_combine_likelihood(func):
    """Split the likelihood calculation across sub state spaces and combine.

    If types are modeled, the data is duplicated for each type. Along with the data, the
    shocks are split across the dense indices.

    """

    @functools.wraps(func)
    def wrapper_distribute_and_combine_likelihood(
        df, base_draws_est, *args, optim_paras, options
    ):
        dense_columns = create_dense_state_space_columns(optim_paras)
        # Duplicate the DataFrame for each type.
        if dense_columns:
            n_obs = df.shape[0]
            n_types = optim_paras["n_types"]

            # Number each state to split the shocks later. This is necessary to keep the
            # regression tests from failing.
            df["__id"] = np.arange(n_obs)
            # Each row of indices corresponds to a state whereas the columns refer to
            # different types.
            indices = np.arange(n_obs * n_types).reshape(n_obs, n_types)

            df_ = pd.concat([df.copy().assign(type=i) for i in range(n_types)])
            splitted_df = _split_dataframe(df_, dense_columns)

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


def _infer_dense_indices_from_arguments(args, kwargs):
    """Infer the dense indices from the arguments.

    This function uses the intersection of all dense indices from the arguments. Since
    the simulated data or data for the likelihood might not comprise all dense
    dimensions, we might need to discard some indices.

    """
    list_of_dense_indices = []
    for arg in args:
        if _is_dictionary_with_tuple_keys(arg):
            list_of_dense_indices.append(set(arg.keys()))
    for kwarg in kwargs.values():
        if _is_dictionary_with_tuple_keys(kwarg):
            list_of_dense_indices.append(set(kwarg.keys()))

    intersection_of_dense_indices = (
        set.intersection(*list_of_dense_indices) if list_of_dense_indices else []
    )

    return intersection_of_dense_indices


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
            args[i] = {idx: arg[idx] for idx in dense_indices}
        else:
            args[i] = {idx: arg for idx in dense_indices}
    for kwarg, value in kwargs.items():
        if _is_dense_dictionary_argument(value, dense_indices):
            kwargs[kwarg] = {idx: kwargs[kwarg][idx] for idx in dense_indices}
        else:
            kwargs[kwarg] = {idx: value for idx in dense_indices}

    # Re-order arguments for zipping.
    args = {idx: [arg[idx] for arg in args] for idx in dense_indices}
    kwargs = {
        idx: {kwarg: value[idx] for kwarg, value in kwargs.items()}
        for idx in dense_indices
    }

    return args, kwargs


def _is_dense_dictionary_argument(argument, dense_indices):
    """Check whether all keys of the dictionary argument are also dense indices.

    We cannot check whether all dense indices are in the argument because `splitted_df`
    in :func:`split_and_combine_df` may not cover all dense combinations.

    """
    return isinstance(argument, dict) and all(idx in argument for idx in dense_indices)


def _split_dataframe(df, dense_columns):
    """Split a DataFrame by creating groups of the same values for the dense dims."""
    groups = {name: group for name, group in df.groupby(dense_columns)}
    groups = convert_dictionary_keys_to_dense_indices(groups)

    return groups


def _split_shocks(base_draws_est, splitted_df, indices, optim_paras):
    """Split the shocks.

    Previously, shocks were assigned to observations which were ordered like observation
    * n_types. Due to the changes to the dense dimensions, this might not be true
    anymore. Thus, ensure the former ordering with the `__id` variable. This will be
    removed with new regression tests.

    """
    splitted_shocks = {}
    for dense_idx, sub_df in splitted_df.items():
        type_ = dense_idx[-1] if optim_paras["n_types"] >= 2 else 0
        sub_indices = sub_df.pop("__id").to_numpy()
        shock_indices_for_group = indices[sub_indices][:, type_].reshape(-1)
        splitted_shocks[dense_idx] = base_draws_est[shock_indices_for_group]

    return splitted_shocks
