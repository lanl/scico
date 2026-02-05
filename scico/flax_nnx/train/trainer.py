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

import time
from typing import Any, Callable, Dict, Optional

import jax
from jax.experimental import mesh_utils
from jax.typing import ArrayLike

import optax

from flax import nnx
from scico.diagnostics import IterationStats

from .checkpoints import checkpoint_restore, checkpoint_save
from .diagnostics import ArgumentStruct, stats_obj
from .input_pipeline import iterate_xy_dataset
from .optax_utils import build_optax_optimizer
from .steps import eval_step, jax_train_step  # , train_step_post
from .typed_dict import ConfigDict, DataSetDict


class BasicFlaxNNXTrainer:
    """Class encapsulating Flax NNX training configuration and execution."""

    def __init__(
        self,
        config: ConfigDict,
        model: Callable,
        train_ds: DataSetDict,
        test_ds: Optional[DataSetDict] = None,
    ):
        """Initializer for :class:`BasicFlaxNNXTrainer`.

        Initializer for :class:`BasicFlaxNNXTrainer` to configure model
        training and evaluation loop. An Optax optimizer is used for
        parameter tuning. A data parallelism strategy is used for training.
        Data is sharded according the number of devices available.

        Args:
            config: Hyperparameter configuration.
            model: Flax nnx model to train.
            train_ds: Dictionary of training data (includes images and
                labels).
            test_ds: Dictionary of testing data (includes images and
                labels). No eval function is run if no test data
                is provided.
        """
        # Store model
        self.model = model

        # Object for storing iteration stats
        self.itstat_object: Optional[IterationStats] = None

        # Configure training loop
        self.set_training_parameters(config)
        self.construct_optimizer(config)

        # Store datasets
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.configure_data_iterators()

        # Create nnx metrics object for registering training stats
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            snr=nnx.metrics.Average("snr"),
        )

        # Create mesh + shardings
        num_devices = jax.local_device_count()
        mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((num_devices,)), ("data",))
        self.model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
        self.data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

        # Store configuration
        self.config = config

    def set_training_parameters(
        self,
        config: ConfigDict,
    ):
        """Extract configuration parameters and construct training functions.

        Parameters and functions are passed in the configuration
        dictionary. Default values are used when parameters are not
        included in configuration.

        Args:
            config: Hyperparameter configuration.
        """
        self.configure_steps(config)
        self.configure_reporting(config)
        self.configure_training_functions(config)

    def configure_steps(
        self,
        config: ConfigDict,
    ):
        """Configure training epochs and frequency for monitoring and checkpointing.

        Args:
            config: Hyperparameter configuration.
        """
        # Set required defaults if not present
        if "batch_size" not in config:
            self.batch_size = 2 * jax.device_count()
        else:
            self.batch_size = config["batch_size"]

        if "num_epochs" not in config:
            self.train_epochs = 10
        else:
            self.train_epochs = config["num_epochs"]

        # Set monitoring frequency or use defaults
        if "checkpoint_every_epochs" not in config:
            self.checkpoint_every_epochs: int = 10
        else:
            self.checkpoint_every_epochs = config["checkpoint_every_epochs"]

        if "log_every_epochs" not in config:
            self.log_every_epochs: int = 5
        else:
            self.log_every_epochs = config["log_every_epochs"]

    def configure_reporting(self, config: ConfigDict):
        """Configure logging and checkpointing.

        The parameters configured correspond to

        - **logflag**: A flag for logging to the output terminal the
              evolution of results. Default: ``False``.
        - **workdir**: Directory to write checkpoints. Default: execution
              directory.
        - **checkpointing**: A flag for checkpointing model state.
              Default: ``False``.

        Args:
            config: Hyperparameter configuration.
        """

        # Determine logging configuration
        if "log" in config:
            self.logflag: bool = config["log"]
            if self.logflag:
                if "stats_obj" in config:
                    self.itstat_object, self.itstat_insert_func = config["stats_obj"]
                else:
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

    def configure_training_functions(self, config: ConfigDict):
        """Construct training functions.

        Default functions are used if not specified in configuration.

        The parameters configured correspond to

        - **criterion**: A function that specifies the loss being minimized
              in training. Default: :meth:`~scico.flax.train.losses.mse_loss`.
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

        if "criterion" in config:
            self.criterion: Callable = config["criterion"]
        else:
            self.criterion = optax.l2_loss

        if "train_step_fn" in config:
            self.train_step_fn: Callable = config["train_step_fn"]
        else:
            self.train_step_fn = jax_train_step

        if "eval_step_fn" in config:
            self.eval_step_fn: Callable = config["eval_step_fn"]
        else:
            self.eval_step_fn = eval_step

        # self.post_lst: Optional[List[Callable]] = None
        # if "post_lst" in config:
        #    self.post_lst = config["post_lst"]

    def configure_data_iterators(
        self,
    ):
        """Configure data iterators.

        Choose between generating (input, label) pairs or only inputs.
        """
        self.dt_iterator_fn: Callable = iterate_xy_dataset
        # Estimate number of batches per epoch
        self.nbatches = self.train_ds["image"].shape[0] // self.batch_size

        self.log_data_snapshot()

    def log_data_snapshot(
        self,
    ):
        """Log datasets information.

        Log information about the loaded training and testing datasets.
        """
        if self.test_ds is not None:
            len_test = self.test_ds["image"].shape[0]
        else:
            len_test = 0

        key = "label"
        # Autoencoder data may only include input (a.k.a. image)
        if key not in self.train_ds.keys():
            key = "image"
        elif self.train_ds["label"].ndim < 3:  # label is class index (not an image)
            key = "image"

        self.log(
            "channels: %d   training signals: %d   testing"
            " signals: %d   signal size: %d\n"
            % (
                self.train_ds[key].shape[-1],  # type: ignore
                self.train_ds[key].shape[0],  # type: ignore
                len_test,
                self.train_ds[key].shape[1],  # type: ignore
            )
        )

    def construct_optimizer(self, config: ConfigDict):
        """Construct Optax optimizer for model training.

        Args:
            config: Hyperparameter configuration.
        """
        tx = build_optax_optimizer(config)
        self.optimizer = nnx.Optimizer(self.model, tx, wrt=nnx.Param)

    def train(self, key: Optional[ArrayLike] = None) -> Optional[IterationStats]:
        """Execute training loop.

        Args:
            key: Key for random generation for models that require
                randomness for training/evaluation (e.g. score).

        Returns:
            Iteration stats object obtained after executing the training
            loop. Note that the iteration stats object is not ``None`` only
            if log is enabled when configuring the training loop.
            The trained model is avaiable in the at
        """
        epochs_offset = 0
        # Handle distributed training using nnx state
        state = nnx.state((self.model, self.optimizer))
        # Before distributing, try to restore if checkpointing is enabled
        if self.checkpointing:
            ok_no_ckpt = True  # It is ok if no checkpoint is found
            state, step = checkpoint_restore(state, self.workdir, ok_no_ckpt)
            if step is not None:
                epochs_offset = step - 1  # > 0 if restarting from checkpoint
        # Distribute state according to model sharding
        state = jax.device_put(state, self.model_sharding)
        nnx.update((self.model, self.optimizer), state)

        # A functional training loop is implemented to reduce Python overhead
        # Split before training loop
        graphdef, state = nnx.split((self.model, self.optimizer, self.metrics))

        # Execute training loop and register stats
        shuffle = True
        key = jax.random.PRNGKey(self.config["seed"])
        t0 = time.time()
        for epoch in range(epochs_offset, self.train_epochs):
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            for batch in self.dt_iterator_fn(
                self.train_ds, self.nbatches, self.batch_size, subkey1, shuffle
            ):
                # Shard data
                x, y = jax.device_put(batch, self.data_sharding)
                # Train
                # self.model.train() # Switch to train mode
                loss, state = self.train_step_fn(graphdef, state, self.criterion, x, y)
            # Update objects after training step
            nnx.update((self.model, self.optimizer, self.metrics), state)
            if (epoch + 1) % self.log_every_epochs == 0:
                self.metrics = self.update_metrics(epoch + 1, self.metrics, t0, subkey3)
            if (epoch + 1) % self.checkpoint_every_epochs == 0:
                self.checkpoint(nnx.state((self.model, self.optimizer)), epoch + 1)

        # Wait for finishing asynchronous execution --> check if this is needed
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

        self.train_time = time.time() - t0

        # Close object for iteration stats if logging
        if self.logflag:
            assert self.itstat_object is not None
            self.itstat_object.end()

        # Extract state from distributed process and update model attribute
        state = nnx.state((self.model, self.optimizer))
        state = jax.device_get(state)
        nnx.update((self.model, self.optimizer), state)

        # Final checkpointing
        self.checkpoint(state, epoch + 1)

        return self.itstat_object  # type: ignore

    def update_metrics(
        self,
        epoch: int,
        metrics,
        t0,
        key: Optional[ArrayLike] = None,
    ):
        """Compute metrics for current model state.

        Metrics for training and testing (eval) sets are computed and
        stored in an iteration stats object. This is executed only if
        logging is enabled.

        Args:
            epoch: Current epoch in training.
            metrics:  Diagnostic statistics computed from
                training set.
            t0: Time when training loop started.
            key: Key for random generation for models that require
                randomness for training/evaluation (e.g. score).
        """
        if not self.logflag:
            return

        summary: Dict[Any] = {}

        # Get current learning rate from optax optimizer (configured to store it).
        summary["train_learning_rate"] = self.optimizer.opt_state.hyperparams["learning_rate"]

        # Log the training metrics
        for metric, value in metrics.compute().items():  # Compute the metrics
            summary[f"train_{metric}"] = value  # Record the metrics
            summary[f"test_{metric}"] = 0.0  # placeholder
        metrics.reset()  # Reset the metrics for the test set

        # Record current epoch and execution time
        summary["epoch"] = epoch
        summary["time"] = time.time() - t0

        if self.test_ds is not None:  # Test data available
            # Eval over testing set
            self.model.eval()  # Switch to eval mode
            shuffle = False
            ntestbatches = self.test_ds["image"].shape[0] // self.batch_size
            for batch in self.dt_iterator_fn(self.test_ds, ntestbatches, self.batch_size, shuffle):
                # Shard data
                x, y = jax.device_put(batch, self.data_sharding)
                # Evaluate
                loss = self.eval_step_fn(self.model, self.criterion, metrics, x, y)

            # Log the test metrics
            for metric, value in metrics.compute().items():
                summary[f"test_{metric}"] = value
            metrics.reset()  # Reset the metrics for the next training epoch

        # Update iteration stats object
        assert isinstance(self.itstat_object, IterationStats)  # for mypy
        self.itstat_object.insert(self.itstat_insert_func(ArgumentStruct(**summary)))

        return metrics

    def checkpoint(self, state, epoch: int):  # pragma: no cover
        """Checkpoint training state if enabled.

        Args:
            state: Flax train state.
            epoch: Current training epoch.
        """
        if self.checkpointing:
            checkpoint_save(state, epoch, self.config, self.workdir)

    def log(self, logstr: str):
        """Print stats to output terminal if logging is enabled.

        Args:
            logstr: String to be logged.
        """
        if self.logflag:
            print(logstr)
