import pytest

import scico.numpy as snp
from scico import functional
from scico.random import randn

INDICATOR = [
    functional.L2BallIndicator,
    functional.NonNegativeIndicator,
    functional.BoxIndicator,
]


@pytest.mark.parametrize("indicator", INDICATOR)
def test_indicator(indicator):
    x, key = randn(shape=(8,), dtype=snp.float32)
    func = indicator()
    assert func(func.prox(x)) == 0.0
