from numba import cgutils
from numba import types
from numba.extending import intrinsic


@intrinsic
def index_tuple_for_array(tyctx, ary, idxary):
    """Converts a an array to a tuple for indexing.

    This function is taken from
    https://gist.github.com/sklam/830fe01343ba95828c3b24c391855c86 to create tuple from
    an array for indexing which is not possible within a Numba function.

    Parameters
    ----------
    ary : np.ndarray
        Array for which the indexer is used.
    idxarray : np.ndarray
        Array which should be converted to a tuple.

    """

    # This is the typing level. Setup the type and constant information here.
    tuple_size = ary.ndim
    typed_tuple = types.UniTuple(dtype=types.intp, count=tuple_size)
    function_signature = typed_tuple(ary, idxary)

    def codegen(cgctx, builder, signature, args):
        # This is the implementation defined using LLVM builder
        lltupty = cgctx.get_value_type(typed_tuple)
        tup = cgutils.get_null_value(lltupty)

        [_, idxaryval] = args

        def array_checker(a):
            if a.size != tuple_size:
                raise IndexError("index array size mismatch")

        # Compile and call array_checker.
        cgctx.compile_internal(builder, array_checker, types.none(idxary), [idxaryval])

        def array_indexer(a, i):
            return a[i]

        # loop to fill the tuple
        for i in range(tuple_size):
            dataidx = cgctx.get_constant(types.intp, i)
            # compile and call array_indexer
            data = cgctx.compile_internal(
                builder,
                array_indexer,
                idxary.dtype(idxary, types.intp),
                [idxaryval, dataidx],
            )
            tup = builder.insert_value(tup, data, i)
        return tup

    return function_signature, codegen
