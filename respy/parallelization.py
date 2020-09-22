"""This module contains the code to control parallel execution."""
import functools

import joblib
import pandas as pd


def parallelize_across_dense_dimensions(func=None, *, n_jobs=1):
    """Parallelizes decorated function across dense state space dimensions.

    Parallelization is only possible if the decorated function has no side-effects to
    other dense dimensions. This might be true for different levels. E.g.
    :meth:`respy.solve._create_choice_rewards` can be directly
    parallelized. :func:`respy.solve._solve_with_backward_induction` cannot be
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
            bypass = kwargs.pop("bypass", {})
            dense_keys = _infer_dense_keys_from_arguments(args, kwargs)

            if dense_keys:
                args_, kwargs_ = _broadcast_arguments(args, kwargs, dense_keys)

                out = joblib.Parallel(n_jobs=n_jobs)(
                    joblib.delayed(func)(*args_[idx], **kwargs_[idx], **bypass)
                    for idx in dense_keys
                )
                # Re-order multiple return values from list of tuples to tuple of lists
                # to tuple of dictionaries to set as state space attributes.
                if isinstance(out[0], tuple):
                    n_returns = len(out[0])
                    tuple_of_lists = tuple(
                        [single_out[i] for single_out in out] for i in range(n_returns)
                    )
                    out = tuple(
                        dict(zip(dense_keys, list_)) for list_ in tuple_of_lists
                    )
                else:
                    out = dict(zip(dense_keys, out))
            else:
                out = func(*args, **kwargs, **bypass)

            return out

        return wrapper_parallelize_across_dense_dimensions

    # Ensures that the decorator can be used without parentheses.
    if callable(func):
        return decorator_parallelize_across_dense_dimensions(func)
    else:
        return decorator_parallelize_across_dense_dimensions


def split_and_combine_df(func):
    """Split the data across dense indices, run a function, and combine again."""

    @functools.wraps(func)
    def wrapper_distribute_and_combine_df(df, *args, **kwargs):
        splitted_df = _split_dataframe(df)
        out = func(splitted_df, *args, **kwargs)
        df = (
            pd.concat(out.values(), sort=False).sort_index()
            if isinstance(out, dict)
            else out
        )

        return df

    return wrapper_distribute_and_combine_df


def _infer_dense_keys_from_arguments(args, kwargs):
    """Infer the dense indices from the arguments.

    Dense indices can be found in dictionaries with only integer keys.

    This function uses the intersection of all dense indices from the arguments. Since
    the simulated data or data for the likelihood might not comprise all dense
    dimensions, we might need to discard some indices.

    """
    list_of_dense_keys = []
    for arg in args:
        if _is_dictionary_with_integer_keys(arg):
            list_of_dense_keys.append(set(arg.keys()))
    for kwarg in kwargs.values():
        if _is_dictionary_with_integer_keys(kwarg):
            list_of_dense_keys.append(set(kwarg.keys()))

    intersection_of_dense_indices = (
        set.intersection(*list_of_dense_keys) if list_of_dense_keys else []
    )

    return intersection_of_dense_indices


def _is_dictionary_with_integer_keys(candidate):
    """Infer whether the argument is a dictionary with integer keys."""
    return (
        isinstance(candidate, dict)
        and len(candidate) > 0
        and all(isinstance(key, int) for key in candidate)
    )


def _broadcast_arguments(args, kwargs, dense_keys):
    """Broadcast arguments to dense state space dimensions."""
    args = list(args) if isinstance(args, tuple) else [args]

    # Broadcast arguments which are not captured in a dictionary with dense state space
    # dimension as keys.
    for i, arg in enumerate(args):
        if _is_dense_dictionary_argument(arg, dense_keys):
            args[i] = {key: arg[key] for key in dense_keys}
        else:
            args[i] = {idx: arg for idx in dense_keys}
    for kwarg, value in kwargs.items():
        if _is_dense_dictionary_argument(value, dense_keys):
            kwargs[kwarg] = {idx: kwargs[kwarg][idx] for idx in dense_keys}
        else:
            kwargs[kwarg] = {idx: value for idx in dense_keys}

    # Re-order arguments for zipping.
    args = {idx: [arg[idx] for arg in args] for idx in dense_keys}
    kwargs = {
        idx: {kwarg: value[idx] for kwarg, value in kwargs.items()}
        for idx in dense_keys
    }

    return args, kwargs


def _is_dense_dictionary_argument(argument, dense_keys):
    """Check whether all keys of the dictionary argument are also dense indices.

    We cannot check whether all dense indices are in the argument because `splitted_df`
    in :func:`split_and_combine_df` may not cover all dense combinations.

    """
    return isinstance(argument, dict) and all(idx in argument for idx in dense_keys)


def _split_dataframe(df):
    """Split a DataFrame by creating groups of the same values for the dense dims."""
    return {name: group for name, group in df.groupby("dense_key")}
