import pytest

from scico.numpy import BlockArray


@pytest.fixture
def ba():
    return


def test_unary():
    x = BlockArray(([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], [42]))
    y = -x

    # TODO FINISH
