from collections import OrderedDict

import pytest

from scico import diagnostics


class TestSet:
    def test_itstat(self):
        its = diagnostics.IterationStats(OrderedDict({"Iter": "%d", "Obj Val": "%8.2e"}))
        its.insert((0, 1.5))
        its.insert((1, 1e2))
        assert its.history()[0].Iter == 0
        assert its.history()[1].Iter == 1
        assert its.history()[1].Obj_Val == 1e2
        assert its.history(transpose=True).Obj_Val == [1.5, 100.0]

    def test_display(self, capsys):
        its = diagnostics.IterationStats({"Iter": "%d"}, display=True, period=2, overwrite=False)
        its.insert((0,))
        cap = capsys.readouterr()
        assert cap.out == "Iter\n----\n   0\n"
        its.insert((1,))
        cap = capsys.readouterr()
        assert cap.out == ""
        its.insert((2,))
        cap = capsys.readouterr()
        assert cap.out == "   2\n"

    def test_exception(self):
        with pytest.raises(TypeError):
            its = diagnostics.IterationStats(["Iter", "%z4d"], display=False)
        with pytest.raises(ValueError):
            its = diagnostics.IterationStats({"Iter": "%z4d"}, display=False)

    def test_warning(self):
        with pytest.warns(UserWarning):
            its = diagnostics.IterationStats({"Iter": "%4e"}, display=False)
