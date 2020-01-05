"""This module contains the code to control parallel execution."""
import copy
import functools

from joblib import delayed
from joblib import Parallel
from joblib import parallel_backend


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
            with parallel_backend("loky"):
                Parallel(n_jobs=1, max_nbytes=None)(
                    delayed(func)(st_sp, *args, **kwargs) for st_sp in state_spaces
                )
        else:
            func(state_space, *args, **kwargs)

    return wrapper_parallelize_across_dense_dimensions
