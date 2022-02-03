# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Parameter tuning using :doc:`ray.tune <ray:tune/index>`."""

import datetime
import os
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import ray

try:
    import ray.tune
except ImportError:
    raise ImportError("Could not import ray.tune; please install it.")
from ray.tune import loguniform, report, uniform  # noqa
from ray.tune.progress_reporter import TuneReporterBase, _get_trials_by_state
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.trial import Trial

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


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
    local_dir: Optional[str] = None,
) -> ray.tune.ExperimentAnalysis:
    """Simplified wrapper for :func:`ray.tune.run`.

    Args:
        run_or_experiment: Function that reports performance values.
        metric: Name of the metric reported in the performance evaluation
            function.
        mode: Either "min" or "max", indicating which represents better
            performance.
        time_budget_s: Maximum time allowed in seconds for the parameter
            search.
        resources_per_trial: A dict mapping keys "cpu" and "gpu" to
            integers specifying the corresponding resources to allocate
            for each performance evaluation trial.
        max_concurrent_trials: Maximum number of trials to run
            concurrently.
        config: Specification of the parameter search space.
        hyperopt: If ``True``, use
            :class:`~ray.tune.suggest.hyperopt.HyperOptSearch` search,
            otherwise use simple random search (see
            :class:`~ray.tune.suggest.basic_variant.BasicVariantGenerator`).
        verbose: Flag indicating whether verbose operation is desired.
            When verbose operation is enabled, the number of pending,
            running, and terminated trials are indicated by "P:", "R:",
            and "T:" respectively, followed by the current best metric
            value and the parameters at which it was reported.
        local_dir: Directory in which to save tuning results. Defaults to
            a subdirectory "ray_results" within the path returned by
            `tempfile.gettempdir()`, corresponding e.g. to
            "/tmp/ray_results" under Linux.

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

    if local_dir is None:
        local_dir = os.path.join(tempfile.gettempdir(), "ray_results")

    return ray.tune.run(
        run_or_experiment,
        metric=metric,
        mode=mode,
        name=name,
        time_budget_s=time_budget_s,
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial=resources_per_trial,
        max_concurrent_trials=max_concurrent_trials,
        reuse_actors=True,
        config=config,
        checkpoint_freq=0,
        **kwargs,
    )
