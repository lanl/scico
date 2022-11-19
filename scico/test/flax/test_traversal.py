import pytest

from flax.core import FrozenDict
from scico.flax.train.traversals import ModelParamTraversal


# Adapted from https://github.com/google/flax/blob/main/tests/traverse_util_test.py
# Traversal are marked as deprecated since Flax v0.4.1. We want to keep ModelParamTraversal functionality
class TestModelParamTraversal:
    def test_only_works_on_model_params(self):
        traversal = ModelParamTraversal(lambda *_: True)
        with pytest.raises(ValueError):
            list(traversal.iterate([]))

    def test_param_selection(self):
        params = {
            "x": {
                "kernel": 1,
                "bias": 2,
                "y": {
                    "kernel": 3,
                    "bias": 4,
                },
                "z": {},
            },
        }
        expected_params = {
            "x": {
                "kernel": 2,
                "bias": 2,
                "y": {
                    "kernel": 6,
                    "bias": 4,
                },
                "z": {},
            },
        }
        names = []

        def filter_fn(name, _):
            names.append(name)  # track names passed to filter_fn for testing
            return "kernel" in name

        traversal = ModelParamTraversal(filter_fn)

        values = list(traversal.iterate(params))
        configs = [
            (params, expected_params),
            (FrozenDict(params), FrozenDict(expected_params)),
        ]
        for model, expected_model in configs:
            assert values == [1, 3]
            new_model = traversal.update(lambda x: x + x, model)
            assert new_model == expected_model
            nm_rhs = ["/x/kernel", "/x/bias", "/x/y/kernel", "/x/y/bias", "/x/z"]
            assert set(names) == set(nm_rhs)
