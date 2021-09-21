from scico import data


class TestSet:
    def test_kodim23_uint(self):
        x = data.kodim23()
        assert x.dtype.name == "uint8"
        assert x.shape == (512, 768, 3)

    def test_kodim23_float(self):
        x = data.kodim23(asfloat=True)
        assert x.dtype.name == "float32"
        assert x.shape == (512, 768, 3)
