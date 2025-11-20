# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Parameter tuning using :doc:`ray.tune <ray:tune/index>`."""

import datetime
import getpass
import logging
import os
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import ray

try:
    import ray.tune

    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
except ImportError:
    raise ImportError("Could not import ray.tune; please install it.")
import ray.air
from ray.tune import (  # noqa
    CheckpointConfig,
    RunConfig,
    Trainable,
    loguniform,
    uniform,
    with_parameters,
)
from ray.tune.experiment.trial import Trial
from ray.tune.progress_reporter import TuneReporterBase, _get_trials_by_state
from ray.tune.result_grid import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch


class _CustomReporter(TuneReporterBase):
    """Custom status reporter for :mod:`ray.tune`."""

    def should_report(self, trials: List[Trial], done: bool = False):
        """Return boolean indicating whether progress should be reported."""
        # Don't report on final call when done to avoid duplicate final output.
        return not done

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        """Report progress across trials."""
        # Get dict of trials in each state.
        trials_by_state = _get_trials_by_state(trials)
        # Construct list of number of trials in each of three possible states.
        num_trials = [len(trials_by_state[state]) for state in ["PENDING", "RUNNING", "TERMINATED"]]
        # Construct string description of number of trials in each state.
        num_trials_str = f"P: {num_trials[0]:3d} R: {num_trials[1]:3d} T: {num_trials[2]:3d} "
        # Get current best trial.
        current_best_trial, metric = self._current_best_trial(trials)
        if current_best_trial is None:
            rslt_str = ""
        else:
            # If current best trial exists, construct string summary
            val = current_best_trial.last_result[metric]
            config = current_best_trial.last_result.get("config", {})
            rslt_str = f" {metric}: {val:.2e} at " + ", ".join(
                [f"{k}: {v:.2e}" for k, v in config.items()]
            )
        # If all trials terminated, print with newline, otherwise carriage return for overwrite
        if num_trials[0] + num_trials[1] == 0:
            end = "\n"
        else:
            end = "\r"
        print(num_trials_str + rslt_str, end=end)


def run(
    run_or_experiment: Union[str, Callable, Type],
    metric: str,
    mode: str,
    time_budget_s: Union[None, int, float, datetime.timedelta] = None,
    num_samples: int = 1,
    resources_per_trial: Union[None, Mapping[str, Union[float, int, Mapping]]] = None,
    max_concurrent_trials: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    hyperopt: bool = True,
    verbose: bool = True,
    storage_path: Optional[str] = None,
) -> ray.tune.ExperimentAnalysis:
    """Simplified wrapper for `ray.tune.run`_.

    .. _ray.tune.run: https://github.com/ray-project/ray/blob/master/python/ray/tune/tune.py#L232

    The `ray.tune.run`_ interface appears to be scheduled for deprecation.
    Use of :class:`Tuner`, which is a simplified interface to
    :class:`ray.tune.Tuner` is recommended instead.

    Args:
        run_or_experiment: Function that reports performance values.
        metric: Name of the metric reported in the performance evaluation
            function.
        mode: Either "min" or "max", indicating which represents better
            performance.
        time_budget_s: Maximum time allowed in seconds for the parameter
            search.
        num_samples: Number of parameter evaluation samples to compute.
        resources_per_trial: A dict mapping keys "cpu" and "gpu" to
            integers specifying the corresponding resources to allocate
            for each performance evaluation trial.
        max_concurrent_trials: Maximum number of trials to run
            concurrently.
        config: Specification of the parameter search space.
        hyperopt: If ``True``, use
            :class:`~ray.tune.search.hyperopt.HyperOptSearch` search,
            otherwise use simple random search (see
            :class:`~ray.tune.search.basic_variant.BasicVariantGenerator`).
        verbose: Flag indicating whether verbose operation is desired.
            When verbose operation is enabled, the number of pending,
            running, and terminated trials are indicated by "P:", "R:",
            and "T:" respectively, followed by the current best metric
            value and the parameters at which it was reported.
        storage_path: Directory in which to save tuning results. Defaults to
            a subdirectory "<username>/ray_results" within the path returned by
            `tempfile.gettempdir()`, corresponding e.g. to
            "/tmp/<username>/ray_results" under Linux.

    Returns:
        Result of parameter search.
    """
    kwargs = {}
    if hyperopt:
        kwargs.update(
            {
                "search_alg": HyperOptSearch(metric=metric, mode=mode),
                "scheduler": AsyncHyperBandScheduler(),
            }
        )
    if verbose:
        kwargs.update({"verbose": 1, "progress_reporter": _CustomReporter()})
    else:
        kwargs.update({"verbose": 0})

    if isinstance(run_or_experiment, str):
        name = run_or_experiment
    else:
        name = run_or_experiment.__name__
    name += "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if storage_path is None:
        try:
            user = getpass.getuser()
        except Exception:  # pragma: no cover
            user = "NOUSER"
        storage_path = os.path.join(tempfile.gettempdir(), user, "ray_results")

    # Record original logger.info
    logger_info = ray.tune.tune.logger.info

    # Replace logger.info with filtered version
    def logger_info_filter(msg, *args, **kwargs):
        if msg[0:15] != "Total run time:":
            logger_info(msg, *args, **kwargs)

    ray.tune.tune.logger.info = logger_info_filter

    result = ray.tune.run(
        run_or_experiment,
        metric=metric,
        mode=mode,
        name=name,
        time_budget_s=time_budget_s,
        num_samples=num_samples,
        storage_path=storage_path,
        resources_per_trial=resources_per_trial,
        max_concurrent_trials=max_concurrent_trials,
        reuse_actors=True,
        config=config,
        checkpoint_freq=0,
        **kwargs,
    )

    # Restore original logger.info
    ray.tune.tune.logger.info = logger_info

    return result


class Tuner(ray.tune.Tuner):
    """Simplified interface for :class:`ray.tune.Tuner`."""

    def __init__(
        self,
        trainable: Union[Type[ray.tune.Trainable], Callable],
        *,
        param_space: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict] = None,
        max_concurrent_trials: Optional[int] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        num_samples: Optional[int] = None,
        num_iterations: Optional[int] = None,
        time_budget: Optional[int] = None,
        reuse_actors: bool = True,
        hyperopt: bool = True,
        verbose: bool = True,
        storage_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
           trainable: Function that reports performance values.
           param_space: Specification of the parameter search space.
           resources: A dict mapping keys "cpu" and "gpu" to integers
              specifying the corresponding resources to allocate for each
              performance evaluation trial.
           max_concurrent_trials: Maximum number of trials to run
            concurrently.
           metric: Name of the metric reported in the performance
              evaluation function.
           mode: Either "min" or "max", indicating which represents
              better performance.
           num_samples: Number of parameter evaluation samples to compute.
           num_iterations: Number of training iterations for evaluation
              of a single configuration. Only required for the Tune Class
              API.
           time_budget: Maximum time allowed in seconds for a single
              parameter evaluation.
           reuse_actors: If ``True``, reuse the same process/object for
              multiple hyperparameters.
           hyperopt: If ``True``, use
              :class:`~ray.tune.search.hyperopt.HyperOptSearch` search,
              otherwise use simple random search (see
              :class:`~ray.tune.search.basic_variant.BasicVariantGenerator`).
           verbose: Flag indicating whether verbose operation is desired.
              When verbose operation is enabled, the number of pending,
              running, and terminated trials are indicated by "P:", "R:",
              and "T:" respectively, followed by the current best metric
              value and the parameters at which it was reported.
           storage_path: Directory in which to save tuning results. Defaults
              to a subdirectory "<username>/ray_results" within the path
              returned by `tempfile.gettempdir()`, corresponding e.g. to
              "/tmp/<username>/ray_results" under Linux.
        """

        k: Any  # Avoid typing errors
        v: Any

        if resources is None:
            trainable_with_resources = trainable
        else:
            trainable_with_resources = ray.tune.with_resources(trainable, resources)

        tune_config = kwargs.pop("tune_config", None)
        tune_config_kwargs = {
            "mode": mode,
            "metric": metric,
            "num_samples": num_samples,
            "reuse_actors": reuse_actors,
        }
        if hyperopt:
            tune_config_kwargs.update(
                {
                    "search_alg": HyperOptSearch(metric=metric, mode=mode),
                    "scheduler": AsyncHyperBandScheduler(),
                }
            )
        if max_concurrent_trials is not None:
            tune_config_kwargs.update({"max_concurrent_trials": max_concurrent_trials})
        if tune_config is None:
            tune_config = ray.tune.TuneConfig(**tune_config_kwargs)
        else:
            for k, v in tune_config_kwargs.items():
                setattr(tune_config, k, v)

        name = trainable.__name__ + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if storage_path is None:
            try:
                user = getpass.getuser()
            except Exception:  # pragma: no cover
                user = "NOUSER"
            storage_path = os.path.join(tempfile.gettempdir(), user, "ray_results")

        run_config = kwargs.pop("run_config", None)
        run_config_kwargs = {"name": name, "storage_path": storage_path, "verbose": 0}
        if verbose:
            run_config_kwargs.update({"verbose": 1, "progress_reporter": _CustomReporter()})
        if num_iterations is not None or time_budget is not None:
            stop_criteria = {}
            if num_iterations is not None:
                stop_criteria.update({"training_iteration": num_iterations})
            if time_budget is not None:
                stop_criteria.update({"time_total_s": time_budget})
            run_config_kwargs.update({"stop": stop_criteria})
        if run_config is None:
            run_config_kwargs.update(
                {"checkpoint_config": CheckpointConfig(checkpoint_at_end=False)}
            )
            run_config = RunConfig(**run_config_kwargs)
        else:
            for k, v in run_config_kwargs.items():
                setattr(run_config, k, v)

        super().__init__(
            trainable_with_resources,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
            **kwargs,
        )

    def fit(self) -> ResultGrid:
        """Initialize ray and call :meth:`ray.tune.Tuner.fit`.

        Initialize ray if not already initialized, and call
        :meth:`ray.tune.Tuner.fit`. If ray was not previously initialized,
        shut it down after fit process has completed.

        Returns:
           Result of parameter search.
        """
        ray_init = ray.is_initialized()
        if not ray_init:
            ray.init(logging_level=logging.ERROR)

        results = super().fit()

        if not ray_init:
            ray.shutdown()

        return results
