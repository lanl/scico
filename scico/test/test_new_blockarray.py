import operator as op

import pytest

from scico.numpy import BlockArray
from scico.numpy.testing import assert_array_equal

for a in dir(op):
    help(getattr(op, a))


@pytest.fixture
def x():
    # any BlockArray, arbitrary shape, content, type
    return BlockArray([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], [42.0]])


@pytest.fixture
def y():
    # another BlockArray, content, type, matching shape
    return BlockArray([[[1.0, 4.0, 6.0], [1.0, 2.0, 3.0]], [-2.0]])


@pytest.mark.parametrize("op", [op.neg, op.pos, op.abs])
def test_unary(op, x):
    actual = op(x)
    expected = BlockArray(op(x_i) for x_i in x)
    assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize(
    "op",
    [
        op.mul,
        op.mod,
        op.lt,
        op.le,
        op.gt,
        op.ge,
        op.floordiv,
        op.eq,
        op.add,
        op.truediv,
        op.sub,
        op.ne,
    ],
)
def test_elementwise_binary(op, x, y):
    actual = op(x, y)
    expected = BlockArray(op(x_i, y_i) for x_i, y_i in zip(x, y))
    assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_not_implemented_binary(x):
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\)"):
        y = x + "a string"


def test_matmul(x):
    # x is ((2, 3), (1,))
    # y is ((3, 1), (1, 2))
    y = BlockArray([[[1.0], [2.0], [3.0]], [[0.0, 1.0]]])
    actual = x @ y
    expected = BlockArray([[[14.0], [0.0]], [0.0, 42.0]])
    assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_property():
    x = BlockArray(([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [0.0]))
    actual = x.shape
    expected = ((2, 3), (1,))
    assert actual == expected


def test_method():
    x = BlockArray(([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], [42.0]))
    actual = x.max()
    expected = BlockArray([[3.0], [42.0]])
    assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
