import os

import pytest

from scico import data

skipif_reason = (
    "\nThe data submodule must be cloned and initialized. If the main repository"
    " is already cloned, use the following in the root directory to get the data"
    " submodule:\n\tgit submodule update --init --recursive\nOtherwise, make sure"
    " to clone using:\n\tgit clone --recurse-submodules git@github.com:lanl/scico.git"
    "\nAnd after cloning run:\n\tgit submodule init && git submodule update.\n"
)

examples = os.path.join(os.path.dirname(data.__file__), "examples")
pytestmark = pytest.mark.skipif(not os.path.isdir(examples), reason=skipif_reason)


class TestSet:
    @pytest.mark.parametrize("options", [(False, "uint8"), (True, "float32")])
    def test_kodim23(self, options):
        asfloat, typestr = options
        img = data.kodim23(asfloat=asfloat)
        assert img.dtype.name == typestr
        assert img.shape == (512, 768, 3)

    @pytest.mark.parametrize("options", [(False, "uint8"), (True, "float32")])
    def test_foam_phantom(self, options):
        asfloat, typestr = options
        vol = data.foam_phantom(asfloat=asfloat)
        assert vol.dtype.name == typestr
        assert vol.shape == (512, 512, 512)
