from collections import OrderedDict

from scico import diagnostics


class TestSet:
    def test_itstat(self):
        its = diagnostics.IterationStats(OrderedDict({"Iter": "%d", "Objective": "%8.2e"}))
        its.insert((0, 1.5))
        its.insert((1, 1e2))
        assert its.history()[0].Iter == 0
        assert its.history()[1].Iter == 1
        assert its.history()[1].Objective == 1e2
        assert its.history(transpose=True).Objective == [1.5, 100.0]
