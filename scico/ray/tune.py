# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Parameter tuning using :mod:`ray.tune`."""

import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import ray
import ray.tune
from ray.tune.progress_reporter import TuneReporterBase, _get_trials_by_state
from ray.tune.trial import Trial

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class _CustomReporter(TuneReporterBase):
    """Custom status reporter for :func:`ray.tune`."""

    def should_report(self, trials: List[Trial], done: bool = False):
        # Don't report on final call when done to avoid duplicate final output.
        return not done

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
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
    verbose: bool = True,
):
    if verbose:
        kwargs = {"verbose": 1, "progress_reporter": _CustomReporter()}
    else:
        kwargs = {"verbose": 0}
    return ray.tune.run(
        run_or_experiment,
        metric=metric,
        mode=mode,
        time_budget_s=time_budget_s,
        num_samples=num_samples,
        resources_per_trial=resources_per_trial,
        max_concurrent_trials=max_concurrent_trials,
        config=config,
        checkpoint_freq=0,
        **kwargs,
    )
