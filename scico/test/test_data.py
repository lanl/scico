import os

import pytest

from scico import data

pytestmark = pytest.mark.skipif(
    len(os.listdir(os.path.abspath("./data"))) == 0,
    reason="\nThe data submodule must be cloned and initialized. If the main repository is already cloned, use the following in the root directory to get the data submodule:\n\tgit submodule update --init --recursive\nOtherwise, make sure to clone using:\n\tgit clone --recurse-submodules git@github.com:lanl/scico.git\nAnd after cloning run:\n\tgit submodule init && git submodule update.\n",
)


class TestSet:
    def test_kodim23_uint(self):
        x = data.kodim23()
        assert x.dtype.name == "uint8"
        assert x.shape == (512, 768, 3)

    def test_kodim23_float(self):
        x = data.kodim23(asfloat=True)
        assert x.dtype.name == "float32"
        assert x.shape == (512, 768, 3)
