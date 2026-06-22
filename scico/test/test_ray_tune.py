import os
import tempfile

import numpy as np

import pytest

try:
    import ray
    from scico.ray import report, tune
except ImportError as e:
    pytest.skip("ray.tune not installed", allow_module_level=True)


def test_random_run():
    def eval_params(config):
        x, y = config["x"], config["y"]
        cost = x**2 + (y - 0.5) ** 2
        report({"cost": cost})

    config = {"x": tune.uniform(-1, 1), "y": tune.uniform(-1, 1)}
    resources = {"gpu": 0, "cpu": 1}
    tune.ray.tune.register_trainable("eval_func", eval_params)
    analysis = tune.run(
        "eval_func",
        metric="cost",
        mode="min",
        num_samples=100,
        config=config,
        resources_per_trial=resources,
        hyperopt=False,
        verbose=False,
        storage_path=os.path.join(tempfile.gettempdir(), "ray_test"),
    )
    best_config = analysis.get_best_config(metric="cost", mode="min")
    assert np.abs(best_config["x"]) < 0.25
    assert np.abs(best_config["y"] - 0.5) < 0.25


def test_random_tune():
    def eval_params(config):
        x, y = config["x"], config["y"]
        cost = x**2 + (y - 0.5) ** 2
        report({"cost": cost})

    config = {"x": tune.uniform(-1, 1), "y": tune.uniform(-1, 1)}
    resources = {"gpu": 0, "cpu": 1}
    tuner = tune.Tuner(
        eval_params,
        param_space=config,
        resources=resources,
        metric="cost",
        mode="min",
        num_samples=100,
        hyperopt=False,
        verbose=False,
        storage_path=os.path.join(tempfile.gettempdir(), "ray_test"),
    )
    results = tuner.fit()
    best_config = results.get_best_result().config
    assert np.abs(best_config["x"]) < 0.25
    assert np.abs(best_config["y"] - 0.5) < 0.25


def test_hyperopt_run():
    def eval_params(config):
        x, y = config["x"], config["y"]
        cost = x**2 + (y - 0.5) ** 2
        report({"cost": cost})

    config = {"x": tune.uniform(-1, 1), "y": tune.uniform(-1, 1)}
    resources = {"gpu": 0, "cpu": 1}
    analysis = tune.run(
        eval_params,
        metric="cost",
        mode="min",
        num_samples=50,
        config=config,
        resources_per_trial=resources,
        hyperopt=True,
        verbose=True,
    )
    best_config = analysis.get_best_config(metric="cost", mode="min")
    assert np.abs(best_config["x"]) < 0.25
    assert np.abs(best_config["y"] - 0.5) < 0.25


def test_hyperopt_tune():
    def eval_params(config):
        x, y = config["x"], config["y"]
        cost = x**2 + (y - 0.5) ** 2
        report({"cost": cost})

    config = {"x": tune.uniform(-1, 1), "y": tune.uniform(-1, 1)}
    resources = {"gpu": 0, "cpu": 1}
    tuner = tune.Tuner(
        eval_params,
        param_space=config,
        resources=resources,
        metric="cost",
        mode="min",
        num_samples=50,
        hyperopt=True,
        verbose=True,
    )
    results = tuner.fit()
    best_config = results.get_best_result().config
    assert np.abs(best_config["x"]) < 0.25
    assert np.abs(best_config["y"] - 0.5) < 0.25


def test_hyperopt_tune_alt_init():
    def eval_params(config):
        x, y = config["x"], config["y"]
        cost = x**2 + (y - 0.5) ** 2
        report({"cost": cost})

    config = {"x": tune.uniform(-1, 1), "y": tune.uniform(-1, 1)}
    tuner = tune.Tuner(
        eval_params,
        param_space=config,
        max_concurrent_trials=4,
        metric="cost",
        mode="min",
        num_samples=50,
        time_budget=2,
        hyperopt=True,
        verbose=True,
        tune_config=ray.tune.TuneConfig(),
        run_config=ray.tune.RunConfig(),
    )
    results = tuner.fit()
    best_config = results.get_best_result().config
    assert np.abs(best_config["x"]) < 0.25
    assert np.abs(best_config["y"] - 0.5) < 0.25
