import scico.numpy as snp

# from scico.random import randn


# from scico.blockarray import BlockArray

x = snp.BlockArray(
    (
        snp.ones((3, 4)),
        snp.arange(4),
    )
)

y = snp.BlockArray(
    (
        2 * snp.ones((3, 4)),
        snp.arange(4),
    )
)

snp.sum(x)

snp.testing.assert_allclose(x, y)
