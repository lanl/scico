# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Integrated access to functionality for training Flax
   models that use only input data (e.g. autoencoders) that can be
   transformed from epoch to epoch (e.g. score models).

Assumes sharded batched data and uses data parallel training.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import time
from functools import partial
from typing import Any, Callable, Dict, Optional

import jax
from jax.typing import ArrayLike

from flax import nnx
from scico.diagnostics import IterationStats
from scico.flax_nnx.train.checkpoints import checkpoint_restore
from scico.flax_nnx.train.diagnostics import ArgumentStruct
from scico.flax_nnx.train.input_pipeline import iterate_x_dataset
from scico.flax_nnx.train.trainer import BasicFlaxNNXTrainer
from scico.flax_nnx.train.typed_dict import ConfigDict, DataSetDict

from .steps import (
    _step_loss,
    _step_t,
    _step_x,
    eval_step_diffusion,
    jax_train_step_diffusion,
)


class FlaxNNXScoreTrainer(BasicFlaxNNXTrainer):
    """Class encapsulating Flax NNX score training configuration and execution."""

    def __init__(
        self,
        config: ConfigDict,
        model: Callable,
        train_ds: DataSetDict,
        test_ds: Optional[DataSetDict] = None,
    ):
        """Initializer for :class:`FlaxNNXScoreTrainer`.

        Initializer for :class:`FlaxNNXScoreTrainer` to configure model
        training and evaluation loop. The trainer assumes that only input
        data is needed for the training, and that this will be modified
        on the fly using other auxiliary processing functions (e.g. step_t,
        step_x, etc.)

        Args:
            config: Hyperparameter configuration.
            model: Flax nnx model to train.
            train_ds: Dictionary of training data (includes images and
                labels).
            test_ds: Dictionary of testing data (includes images and
                labels). No eval function is run if no test data
                is provided.
        """
        # Configure default train diffusion functions
        if "train_step_fn" not in config:
            config["train_step_fn"] = jax_train_step_diffusion

        if "eval_step_fn" not in config:
            config["eval_step_fn"] = eval_step_diffusion

        super().__init__(config, model, train_ds, test_ds)

        # Configure processing for training loop
        self.set_data_processing_functions(config)

        # Create nnx metrics object for registering training stats
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

    def set_data_processing_functions(
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

        if "step_t" in config:
            self.step_t: Callable = config["step_t"]
        else:
            self.step_t = _step_t

        if "step_x" in config:
            self.step_x: Callable = config["step_x"]
        else:
            self.step_x = partial(_step_x, stddev_prior=self.config["stddev_prior"])

        if "step_loss" in config:
            self.step_loss: Callable = config["step_loss"]
        else:
            self.step_loss = _step_loss

    def configure_data_iterators(
        self,
    ):
        """Configure data iterators.

        Generate training data set only from inputs.

        """
        self.dt_iterator_fn: Callable = iterate_x_dataset

        # Estimate number of batches per epoch
        self.nbatches = self.train_ds["image"].shape[0] // self.batch_size

        self.log_data_snapshot()

    def train(self, key: Optional[ArrayLike] = None) -> Optional[IterationStats]:
        """Execute training loop.

        Args:
            key: Key for random generation for models that require
                randomness for training/evaluation (e.g. score).

        Returns:
            Iteration stats object obtained after executing the training
            loop. Note that the iteration stats object is not ``None``
            only if log is enabled when configuring the training loop.
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
                x = jax.device_put(batch, self.data_sharding)
                # Train
                # self.model.train() # Switch to train mode
                subkey2, step_key1, step_key2 = jax.random.split(subkey2, 3)
                t, batch_t = self.step_t(x, step_key1)
                z, batch_std, batch_x = self.step_x(x, step_key2, t)

                loss, state = self.train_step_fn(
                    graphdef, state, self.criterion, self.step_loss, batch_x, batch_t, z, batch_std
                )

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
                x = jax.device_put(batch, self.data_sharding)
                key, step_key1, step_key2 = jax.random.split(key, 3)
                t, batch_t = self.step_t(x, step_key1)
                z, batch_std, batch_x = self.step_x(x, step_key2, t)
                # Evaluate
                loss = self.eval_step_fn(
                    self.model,
                    self.criterion,
                    metrics,
                    self.step_loss,
                    batch_x,
                    batch_t,
                    z,
                    batch_std,
                )

            # Log the test metrics
            for metric, value in metrics.compute().items():
                summary[f"test_{metric}"] = value
            metrics.reset()  # Reset the metrics for the next training epoch

        # Update iteration stats object
        assert isinstance(self.itstat_object, IterationStats)  # for mypy
        self.itstat_object.insert(self.itstat_insert_func(ArgumentStruct(**summary)))

        return metrics
