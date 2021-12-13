import numpy as np

import pytest

try:
    import ray
    from scico.ray import tune

    ray.init(local_mode=True)
except ImportError as e:
    pytest.skip("ray.tune not installed", allow_module_level=True)


def eval_params(config):
    x, y = config["x"], config["y"]
    cost = x ** 2 + (y - 0.5) ** 2
    tune.report(cost=cost)


config = {"x": tune.uniform(-1, 1), "y": tune.uniform(-1, 1)}
resources = {"gpu": 0, "cpu": 1}


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_random():
    analysis = tune.run(
        eval_params,
        metric="cost",
        mode="min",
        num_samples=100,
        config=config,
        resources_per_trial=resources,
        hyperopt=False,
        verbose=False,
    )
    best_config = analysis.get_best_config(metric="cost", mode="min")
    assert np.abs(best_config["x"]) < 0.25
    assert np.abs(best_config["y"] - 0.5) < 0.25


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_hyperopt():
    analysis = tune.run(
        eval_params,
        metric="cost",
        mode="min",
        num_samples=100,
        config=config,
        resources_per_trial=resources,
        hyperopt=True,
        verbose=True,
    )
    best_config = analysis.get_best_config(metric="cost", mode="min")
    assert np.abs(best_config["x"]) < 0.25
    assert np.abs(best_config["y"] - 0.5) < 0.25
