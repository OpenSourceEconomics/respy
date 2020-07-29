import numba as nb
import pytest
from numba.typed import Dict

from respy.parallelization import _infer_dense_keys_from_arguments
from respy.parallelization import _is_dense_dictionary_argument
from respy.parallelization import _is_dictionary_with_integer_keys


def _typeddict_wo_integer_keys():
    dictionary = Dict.empty(
        key_type=nb.types.UniTuple(nb.types.int64, 2), value_type=nb.types.int64
    )
    dictionary[(1, 2)] = 1
    return dictionary


def _typeddict_w_integer_keys():
    dictionary = Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)
    dictionary[1] = 1
    return dictionary


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_, expected",
    [
        ({1: 2, 3: 4}, True),
        (1, False),
        ([3, 4, 5], False),
        (_typeddict_wo_integer_keys(), False),
        (_typeddict_w_integer_keys(), False),
    ],
)
def test_is_dictionary_with_integer_keys(input_, expected):
    assert _is_dictionary_with_integer_keys(input_) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "args, kwargs, expected",
    [(({1: None, 2: None},), {"kwarg_1": {2: None, 3: None}}, {2})],
)
def test_infer_dense_keys_from_arguments(args, kwargs, expected):
    result = _infer_dense_keys_from_arguments(args, kwargs)
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "arg, dense_keys, expected",
    [({1: None, 2: None}, {1, 2}, True), ((1,), {1, 2, 3}, False)],
)
def test_is_dense_dictionary_argument(arg, dense_keys, expected):
    result = _is_dense_dictionary_argument(arg, dense_keys)
    assert result is expected
