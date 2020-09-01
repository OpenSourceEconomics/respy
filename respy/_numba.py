"""Special functions for using numba."""
import warnings

import numba as nb
import numpy as np
from numba import NumbaDeprecationWarning
from numba import types
from numba.extending import intrinsic

# Fix for transition to Numba 0.50. cgutils was moved from numba.cgutils to
# numba.core.cgutils.
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaDeprecationWarning)
        from numba import cgutils
except ImportError:
    from numba.core import cgutils


@intrinsic  # noqa: U100
def array_to_tuple(tyctx, array_or_dict, indexer_array):  # noqa: U100
    """Convert an array to a tuple for indexing.

    This function is taken from
    https://gist.github.com/sklam/830fe01343ba95828c3b24c391855c86 to create tuple from
    an array for indexing which is not possible within a Numba function.

    Parameters
    ----------
    array_or_dict : numpy.ndarray or numba.typed.Dict
        Array for which the indexer is used.
    indexer_array : numpy.ndarray
        Array which should be converted to a tuple.

    """
    # This is the typing level. Setup the type and constant information here.
    if isinstance(array_or_dict, types.DictType):
        tuple_size = len(array_or_dict.key_type)
    else:
        tuple_size = array_or_dict.ndim

    tuple_type = indexer_array.dtype
    typed_tuple = types.UniTuple(dtype=tuple_type, count=tuple_size)
    function_signature = typed_tuple(array_or_dict, indexer_array)

    def codegen(cgctx, builder, signature, args):
        # This is the implementation defined using LLVM builder.
        lltupty = cgctx.get_value_type(typed_tuple)
        tup = cgutils.get_null_value(lltupty)

        [_, idxaryval] = args

        def array_checker(a):
            if a.size != tuple_size:
                raise IndexError("index array size mismatch")

        # Compile and call array_checker.
        cgctx.compile_internal(
            builder, array_checker, types.none(indexer_array), [idxaryval]
        )

        def array_indexer(a, i):
            return a[i]

        # loop to fill the tuple
        for i in range(tuple_size):
            dataidx = cgctx.get_constant(types.intp, i)
            # compile and call array_indexer
            data = cgctx.compile_internal(
                builder,
                array_indexer,
                indexer_array.dtype(indexer_array, types.intp),
                [idxaryval, dataidx],
            )
            tup = builder.insert_value(tup, data, i)
        return tup

    return function_signature, codegen


@nb.njit
def sum_over_numba_boolean_unituple(tuple_):
    """Compute the sum over a boolean :class:`numba.types.UniTuple`.

    Parameters
    ----------
    tuple_ : numba.types.UniTuple[bool]

    Returns
    -------
    sum_ : Union[float, int]

    """
    return np.sum(np.array([1 for i in tuple_ if i]))
