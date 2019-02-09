"""This module contains fast routines or replacements of numpy functions."""
import numpy as np
from numba import njit


@njit
def clip(a, a_min=None, a_max=None):
    """Replacement for np.clip as long it is not supported by Numba.

    Mirrors functionality from
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html.
    Support is currently planned for Numba version 0.43.0. The PR on Github is
    https://github.com/numba/numba/pull/3468.

    Warning
    -------
    Scalars are not supported as in many Numba functions. Check out
    https://github.com/numba/numba/issues/3175 and other issues.

    Example
    -------
    >>> import numpy as np
    >>> a = np.arange(-10, 10)
    >>> assert np.allclose(clip(a, -5, 5), np.clip(a, -5, 5))

    """
    a_min_is_none = a_min is None
    a_max_is_none = a_max is None

    if a_min_is_none and a_max_is_none:
        raise ValueError("array_clip: must set either max or min")

    elif a_min_is_none:
        ret = np.empty_like(a)
        for index, val in np.ndenumerate(a):
            if val > a_max:
                ret[index] = a_max
            else:
                ret[index] = val
        return ret

    elif a_max_is_none:
        ret = np.empty_like(a)
        for index, val in np.ndenumerate(a):
            if val < a_min:
                ret[index] = a_min
            else:
                ret[index] = val
        return ret

    else:
        ret = np.empty_like(a)
        for index, val in np.ndenumerate(a):
            if val < a_min:
                ret[index] = a_min
            elif val > a_max:
                ret[index] = a_max
            else:
                ret[index] = val
        return ret


@njit
def triu_indices(n, k=0, m=None):
    """Replacement for np.triu_indices as long it is not supported by Numba.

    Mirrors functionality from
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.triu_indices.html.

    Examples
    --------
    >>> import numpy as np
    >>> assert np.allclose(triu_indices(2), np.triu_indices(2))

    """
    if k != 0:
        raise NotImplementedError("Diagonal offset is not implemented.")

    if m is None:
        m = n

    num_elements_in_triangle = tri_n_with_diag(n)
    rows = np.zeros(num_elements_in_triangle, dtype=np.int8)
    cols = np.zeros(num_elements_in_triangle, dtype=np.int8)

    i = 0
    for row in range(n):
        for col in range(m):
            if row <= col:
                rows[i], cols[i] = row, col
                i += 1
            else:
                continue

    return rows, cols


@njit
def tril_indices(n, k=0, m=None):
    """Replacement for np.triu_indices as long it is not supported by Numba.

    Mirrors functionality from
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.triu_indices.html.

    Examples
    --------
    >>> import numpy as np
    >>> assert np.allclose(tril_indices(2), np.tril_indices(2))

    """
    if k != 0:
        raise NotImplementedError("Diagonal offset is not implemented.")

    if m is None:
        m = n

    num_elements_in_triangle = tri_n_with_diag(n)
    rows = np.zeros(num_elements_in_triangle, dtype=np.int8)
    cols = np.zeros(num_elements_in_triangle, dtype=np.int8)

    i = 0
    for row in range(n):
        for col in range(m):
            if col <= row:
                rows[i], cols[i] = row, col
                i += 1
            else:
                continue

    return rows, cols


@njit
def tri_n(n):
    """Number of elements in matrix triangle."""
    return n * (n - 1) // 2


@njit
def tri_n_with_diag(n):
    """Number of elements in matrix triangle + diagonal."""
    return tri_n(n) + n


@njit
def diagonal(a):
    """Return elements of the matrix diagonal.

    Parameters
    ----------
    a : np.array
        Matrix

    Returns
    -------
    np.array
        Array containing elements of the diagonal.

    Example
    -------
    >>> import numpy as np
    >>> a = np.arange(16).reshape(4, 4)
    >>> assert np.allclose(diagonal(a), np.diagonal(a))

    """
    dim = a.shape[0]

    out = np.empty(dim)

    for i in range(dim):
        out[i] = a[i, i]

    return out


@njit
def count_nonzero(a):
    """Return number of non-zero elements.

    Parameters
    ----------
    a : np.array

    Returns
    -------
    int
        Number of non-zero elements.

    Example
    -------
    >>> import numpy as np
    >>> a = np.arange(-8, 8).reshape(4, 4)
    >>> np.allclose(count_nonzero(a), np.count_nonzero(a))

    """
    return np.sum(0 < a)
