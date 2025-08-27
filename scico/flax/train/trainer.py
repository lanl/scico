# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Class providing integrated access to functionality for training Flax
   models.

Assumes sharded batched data and uses data parallel training.
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
from jax import lax

from flax import jax_utils
from flax.training import common_utils
from scico.diagnostics import IterationStats
from scico.numpy import Array

from .checkpoints import checkpoint_restore, checkpoint_save
from .clu_utils import get_parameter_overview
from .diagnostics import ArgumentStruct, compute_metrics, stats_obj
from .input_pipeline import create_input_iter
from .learning_rate import create_cnst_lr_schedule
from .losses import mse_loss
from .state import TrainState, create_basic_train_state
from .steps import eval_step, train_step, train_step_post
from .typed_dict import ConfigDict, DataSetDict, MetricsDict, ModelVarDict

ModuleDef = Any
KeyArray = Union[Array, jax.Array]
PyTree = Any
DType = Any


# sync across replicas
def sync_batch_stats(state: TrainState) -> TrainState:
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch
    # statistics and those are synced before evaluation
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


# pmean only works inside pmap because it needs an axis name.
#: This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


class BasicFlaxTrainer:
    """Class encapsulating Flax training configuration and execution."""

    def __init__(
        self,
        config: ConfigDict,
        model: ModuleDef,
        train_ds: DataSetDict,
        test_ds: DataSetDict,
        variables0: Optional[ModelVarDict] = None,
    ):
        """Initializer for :class:`BasicFlaxTrainer`.

        Initializer for :class:`BasicFlaxTrainer` to configure model
        training and evaluation loop. Construct a Flax train state (which
        includes the model apply function, the model parameters and an
        Optax optimizer). This uses data parallel training assuming
        sharded batched data.

        Args:
            config: Hyperparameter configuration.
            model: Flax model to train.
            train_ds: Dictionary of training data (includes images and
                labels).
            test_ds: Dictionary of testing data (includes images and
                labels).
            variables0: Optional initial state of model parameters.
        """
        # Configure seed
        if "seed" not in config:
            key = jax.random.PRNGKey(0)
        else:
            key = jax.random.PRNGKey(config["seed"])
        # Split seed for data iterators and model initialization
        key1, key2 = jax.random.split(key)

        # Object for storing iteration stats
        self.itstat_object: Optional[IterationStats] = None

        # Configure training loop
        len_train = train_ds["image"].shape[0]
        len_test = test_ds["image"].shape[0]
        self.set_training_parameters(config, len_train, len_test)
        self.construct_data_iterators(train_ds, test_ds, key1, model.dtype)

        self.define_parallel_training_functions()

        self.initialize_training_state(config, key2, model, variables0)

        # Store configuration
        self.config = config

    def set_training_parameters(
        self,
        config: ConfigDict,
        len_train: int,
        len_test: int,
    ):
        """Extract configuration parameters and construct training functions.

        Parameters and functions are passed in the configuration
        dictionary. Default values are used when parameters are not
        included in configuration.

        Args:
            config: Hyperparameter configuration.
            len_train: Number of samples in training set.
            len_test: Number of samples in testing set.
        """
        self.configure_steps(config, len_train, len_test)
        self.configure_reporting(config)
        self.configure_training_functions(config)

    def configure_steps(
        self,
        config: ConfigDict,
        len_train: int,
        len_test: int,
    ):
        """Configure training, evaluation and monitoring steps.

        Args:
            config: Hyperparameter configuration.
            len_train: Number of samples in training set.
            len_test: Number of samples in testing set.
        """
        # Set required defaults if not present
        if "batch_size" not in config:
            batch_size = 2 * jax.device_count()
        else:
            batch_size = config["batch_size"]
        if "num_epochs" not in config:
            num_epochs = 10
        else:
            num_epochs = config["num_epochs"]

        # Determine sharded vs. batch partition
        if batch_size % jax.device_count() > 0:
            raise ValueError("Batch size must be divisible by the number of devices.")
        self.local_batch_size: int = batch_size // jax.process_count()

        # Training steps
        self.steps_per_epoch: int = len_train // batch_size
        config["steps_per_epoch"] = self.steps_per_epoch  # needed for creating lr schedule
        self.num_steps: int = int(self.steps_per_epoch * num_epochs)

        # Evaluation (over testing set) steps
        num_validation_examples: int = len_test
        if "steps_per_eval" not in config:
            self.steps_per_eval: int = num_validation_examples // batch_size
        else:
            self.steps_per_eval = config["steps_per_eval"]

        # Determine monitoring steps
        if "steps_per_checkpoint" not in config:
            self.steps_per_checkpoint: int = self.steps_per_epoch * 10
        else:
            self.steps_per_checkpoint = config["steps_per_checkpoint"]

        if "log_every_steps" not in config:
            self.log_every_steps: int = self.steps_per_epoch * 20
        else:
            self.log_every_steps = config["log_every_steps"]

    def configure_reporting(self, config: ConfigDict):
        """Configure logging and checkpointing.

        The parameters configured correspond to

        - **logflag**: A flag for logging to the output terminal the
              evolution of results. Default: ``False``.
        - **workdir**: Directory to write checkpoints. Default: execution
              directory.
        - **checkpointing**: A flag for checkpointing model state.
              Default: ``False``.
        - **return_state**: A flag for returning the train state instead
              of the model variables. Default: ``False``, i.e. return
              model variables.

        Args:
            config: Hyperparameter configuration.
        """

        # Determine logging configuration
        if "log" in config:
            self.logflag: bool = config["log"]
            if self.logflag:
                self.itstat_object, self.itstat_insert_func = stats_obj()
        else:
            self.logflag = False

        # Determine checkpointing configuration
        if "workdir" in config:
            self.workdir: str = config["workdir"]
        else:
            self.workdir = "./"

        if "checkpointing" in config:
            self.checkpointing: bool = config["checkpointing"]
        else:
            self.checkpointing = False

        # Determine variable to return at end of training
        if "return_state" in config:
            # Returning Flax train state
            self.return_state = config["return_state"]
        else:
            # Return model variables
            self.return_state = False

    def configure_training_functions(self, config: ConfigDict):
        """Construct training functions.

        Default functions are used if not specified in configuration.

        The parameters configured correspond to

        - **lr_schedule**: A function that creates an Optax learning rate
              schedule. Default: :meth:`~scico.flax.train.learning_rate.create_cnst_lr_schedule`.
        - **criterion**: A function that specifies the loss being minimized
              in training. Default: :meth:`~scico.flax.train.losses.mse_loss`.
        - **create_train_state**: A function that creates a Flax train state
              and initializes it. A train state object helps to keep optimizer
              and module functionality grouped for training. Default:
              :meth:`~scico.flax.train.state.create_basic_train_state`.
        - **train_step_fn**: A function that executes a training step.
              Default: :meth:`~scico.flax.train.steps.train_step`, i.e.
              use the standard train step.
        - **eval_step_fn**: A function that executes an eval step. Default:
              :meth:`~scico.flax.train.steps.eval_step`, i.e. use the
              standard eval step.
        - **metrics_fn**: A function that computes metrics. Default:
              :meth:`~scico.flax.train.diagnostics.compute_metrics`, i.e.
              use the standard compute metrics function.
        - **post_lst**: List of postprocessing functions to apply to
              parameter set after optimizer step (e.g. clip to a specified
              range, normalize, etc.).

        Args:
            config: Hyperparameter configuration.
        """

        if "lr_schedule" in config:
            create_lr_schedule: Callable = config["lr_schedule"]
            self.lr_schedule = create_lr_schedule(config)
        else:
            self.lr_schedule = create_cnst_lr_schedule(config)

        if "criterion" in config:
            self.criterion: Callable = config["criterion"]
        else:
            self.criterion = mse_loss

        if "create_train_state" in config:
            self.create_train_state: Callable = config["create_train_state"]
        else:
            self.create_train_state = create_basic_train_state

        if "train_step_fn" in config:
            self.train_step_fn: Callable = config["train_step_fn"]
        else:
            self.train_step_fn = train_step

        if "eval_step_fn" in config:
            self.eval_step_fn: Callable = config["eval_step_fn"]
        else:
            self.eval_step_fn = eval_step

        if "metrics_fn" in config:
            self.metrics_fn: Callable = config["metrics_fn"]
        else:
            self.metrics_fn = compute_metrics

        self.post_lst: Optional[List[Callable]] = None
        if "post_lst" in config:
            self.post_lst = config["post_lst"]

    def construct_data_iterators(
        self,
        train_ds: DataSetDict,
        test_ds: DataSetDict,
        key: KeyArray,
        mdtype: DType,
    ):
        """Construct iterators for training and testing (evaluation) sets.

        Args:
            train_ds: Dictionary of training data (includes images
                and labels).
            test_ds: Dictionary of testing data (includes images
                and labels).
            key: A PRNGKey used as the random key.
            mdtype: Output type of Flax model to be trained.
        """
        size_device_prefetch = 2  # Set for GPU

        self.train_dt_iter = create_input_iter(
            key,
            train_ds,
            self.local_batch_size,
            size_device_prefetch,
            mdtype,
            train=True,
        )
        self.eval_dt_iter = create_input_iter(
            key,  # eval: no permutation
            test_ds,
            self.local_batch_size,
            size_device_prefetch,
            mdtype,
            train=False,
        )

        self.ishape = train_ds["image"].shape[1:3]
        self.log(
            "channels: %d   training signals: %d   testing"
            " signals: %d   signal size: %d\n"
            % (
                train_ds["label"].shape[-1],
                train_ds["label"].shape[0],
                test_ds["label"].shape[0],
                train_ds["label"].shape[1],
            )
        )

    def define_parallel_training_functions(self):
        """Construct parallel versions of training functions.

        Construct parallel versions of training functions via
        :func:`jax.pmap`.
        """
        if self.post_lst is not None:
            self.p_train_step = jax.pmap(
                functools.partial(
                    train_step_post,
                    train_step_fn=self.train_step_fn,
                    learning_rate_fn=self.lr_schedule,
                    criterion=self.criterion,
                    metrics_fn=self.metrics_fn,
                    post_lst=self.post_lst,
                ),
                axis_name="batch",
            )
        else:
            self.p_train_step = jax.pmap(
                functools.partial(
                    self.train_step_fn,
                    learning_rate_fn=self.lr_schedule,
                    criterion=self.criterion,
                    metrics_fn=self.metrics_fn,
                ),
                axis_name="batch",
            )
        self.p_eval_step = jax.pmap(
            functools.partial(
                self.eval_step_fn, criterion=self.criterion, metrics_fn=self.metrics_fn
            ),
            axis_name="batch",
        )

    def initialize_training_state(
        self,
        config: ConfigDict,
        key: KeyArray,
        model: ModuleDef,
        variables0: Optional[ModelVarDict] = None,
    ):
        """Construct and initialize Flax train state.

        A train state object helps to keep optimizer and module
        functionality grouped for training.

        Args:
            config: Hyperparameter configuration.
            key: A PRNGKey used as the random key.
            model: Flax model to train.
            variables0: Optional initial state of model parameters.
        """
        # Create Flax training state
        state = self.create_train_state(
            key, config, model, self.ishape, self.lr_schedule, variables0
        )
        # Only restore if no initialization is provided
        if self.checkpointing and variables0 is None:
            ok_no_ckpt = True  # It is ok if no checkpoint is found
            state = checkpoint_restore(state, self.workdir, ok_no_ckpt)

        self.log("Network Structure:")
        self.log(get_parameter_overview(state.params) + "\n")
        if hasattr(state, "batch_stats"):
            self.log("Batch Normalization:")
            self.log(get_parameter_overview(state.batch_stats) + "\n")

        self.state = state

    def train(self) -> Tuple[Dict[str, Any], Optional[IterationStats]]:
        """Execute training loop.

        Returns:
            Model variables extracted from :class:`.TrainState` and
            iteration stats object obtained after executing the training
            loop. Alternatively the :class:`.TrainState` can be returned
            directly instead of the model variables. Note that the
            iteration stats object is not ``None`` only if log is enabled
            when configuring the training loop.
        """
        state = self.state
        step_offset = int(state.step)  # > 0 if restarting from checkpoint

        # For parallel training
        state = jax_utils.replicate(state)
        # Execute training loop and register stats
        t0 = time.time()
        self.log("Initial compilation, which might take some time ...")

        train_metrics: List[Any] = []

        for step, batch in zip(range(step_offset, self.num_steps), self.train_dt_iter):
            state, metrics = self.p_train_step(state, batch)
            # Training metrics computed in step
            train_metrics.append(metrics)
            if step == step_offset:
                self.log("Initial compilation completed.\n")
            if (step + 1) % self.log_every_steps == 0:
                # sync batch statistics across replicas
                state = sync_batch_stats(state)
                self.update_metrics(state, step, train_metrics, t0)
                train_metrics = []
            if (step + 1) % self.steps_per_checkpoint == 0 or step + 1 == self.num_steps:
                # sync batch statistics across replicas
                state = sync_batch_stats(state)
                self.checkpoint(state)

        # Wait for finishing asynchronous execution
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        # Close object for iteration stats if logging
        if self.logflag:
            assert self.itstat_object is not None
            self.itstat_object.end()

        state = sync_batch_stats(state)
        # Final checkpointing
        self.checkpoint(state)
        # Extract one copy of state
        state = jax_utils.unreplicate(state)
        if self.return_state:
            return state, self.itstat_object  # type: ignore

        dvar: ModelVarDict = {
            "params": state.params,
            "batch_stats": state.batch_stats,
        }

        self.train_time = time.time() - t0

        return dvar, self.itstat_object  # type: ignore

    def update_metrics(self, state: TrainState, step: int, train_metrics: List[MetricsDict], t0):
        """Compute metrics for current model state.

        Metrics for training and testing (eval) sets are computed and
        stored in an iteration stats object. This is executed only if
        logging is enabled.

        Args:
            state: Flax train state which includes the model apply
                function and the model parameters.
            step: Current step in training.
            train_metrics: List of diagnostic statistics computed from
                training set.
            t0: Time when training loop started.
        """
        if not self.logflag:
            return

        eval_metrics: List[Any] = []

        # Build summary dictionary for logging
        # Include training stats
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f"train_{k}": v
            for k, v in jax.tree_util.tree_map(lambda x: x.mean(), train_metrics).items()
        }
        epoch = step // self.steps_per_epoch
        summary["epoch"] = epoch
        summary["time"] = time.time() - t0

        # Eval over testing set
        for _ in range(self.steps_per_eval):
            eval_batch = next(self.eval_dt_iter)
            metrics = self.p_eval_step(state, eval_batch)
            eval_metrics.append(metrics)
        # Compute testing metrics
        eval_metrics = common_utils.get_metrics(eval_metrics)

        # Add testing stats to summary
        summary_eval = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
        summary.update(summary_eval)

        # Update iteration stats object
        assert isinstance(self.itstat_object, IterationStats)  # for mypy
        self.itstat_object.insert(self.itstat_insert_func(ArgumentStruct(**summary)))

    def checkpoint(self, state: TrainState):  # pragma: no cover
        """Checkpoint training state if enabled.

        Args:
            state: Flax train state.
        """
        if self.checkpointing:
            checkpoint_save(jax_utils.unreplicate(state), self.config, self.workdir)

    def log(self, logstr: str):
        """Print stats to output terminal if logging is enabled.

        Args:
            logstr: String to be logged.
        """
        if self.logflag:
            print(logstr)
