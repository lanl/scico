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

from functools import partial
from typing import Callable, Optional

import jax
from jax.typing import ArrayLike

from flax import nnx
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
        # Connect data iterators with train/eval steps (for data sharding)
        self.one_train_epoch_fn: Callable = self.one_train_epoch_x
        self.one_eval_epoch_fn: Callable = self.one_eval_epoch_x

        # Estimate number of batches per epoch
        self.nbatches = self.train_ds["image"].shape[0] // self.batch_size

        self.log_data_snapshot()

    def one_train_epoch_x(self, graphdef, state, key: ArrayLike):
        """Function that defines one epoch of training for a VAE that uses only input data.

        This function iterates over the whole training set using randomly shuffled batches.

        Args:
            graphdef: Graph representation of model.
            state: NNX state object including pytrees for all the
                model and optimizer graph nodes.
            key: Key for random generation in VAE forward pass.

        Returns:
            Training loss and updated state and key.
        """
        shuffle = True
        key, subkey1, subkey2 = jax.random.split(key, 3)
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
        return loss, state, key

    def one_eval_epoch_x(self, metrics, key):
        """Function that defines one epoch of evaluation for a VAE that uses only input data.

        This function iterates over the whole testing set using sequential batches.

        Args:
            metrics: Dictionary of metrics to evaluate.
            key: Key for random generation in VAE forward pass.

        Returns:
            Testing loss and updated metrics and key.
        """
        self.model.eval()  # Switch to eval mode
        shuffle = False
        ntestbatches = self.test_ds["image"].shape[0] // self.batch_size
        for batch in self.dt_iterator_fn(self.test_ds, ntestbatches, self.batch_size, shuffle):
            # Shard data
            x = jax.device_put(batch, self.data_sharding)
            # Eval step
            key, step_key1, step_key2 = jax.random.split(key, 3)
            t, batch_t = self.step_t(x, step_key1)
            z, batch_std, batch_x = self.step_x(x, step_key2, t)
            loss = self.eval_step_fn(
                self.model, self.criterion, self.step_loss, metrics, batch_x, batch_t, z, batch_std
            )
        return loss, metrics, key
