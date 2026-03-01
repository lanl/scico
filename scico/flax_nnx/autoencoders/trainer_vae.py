# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Integrated access to functionality for training Flax NNX
   models that require random generation during forward
   passes (e.g. variational autoencoders).

Assumes sharded batched data and uses data parallel training.
"""
from typing import Callable, Optional

import jax
from jax.typing import ArrayLike

from flax import nnx
from scico.flax_nnx.train.trainer import BasicFlaxNNXTrainer
from scico.flax_nnx.train.typed_dict import ConfigDict, DataSetDict

from .steps import _kl_loss_fn, eval_step_vae, jax_train_step_vae


class FlaxNNXVAETrainer(BasicFlaxNNXTrainer):
    """Class encapsulating Flax NNX variational autoencoder (VAE) training configuration and execution."""

    def __init__(
        self,
        config: ConfigDict,
        model: Callable,
        train_ds: DataSetDict,
        test_ds: Optional[DataSetDict] = None,
    ):
        """Initializer for :class:`FlaxNNXVAETrainer`.

        Initializer for :class:`FlaxNNXVAETrainer` to configure model
        training and evaluation loop. The trainer assumes that only input
        data is needed for the training. Also, if conditioning this will be modified
        on the fly using other auxiliary processing functions (e.g. step_t,
        step_x, etc.)

        Args:
            config: Hyperparameter configuration.
            model: Flax nnx model to train.
            train_ds: Dictionary of training data (includes images and
                may include labels).
            test_ds: Dictionary of testing data (includes images and
                may include labels). No eval function is run if no test data
                is provided.
        """
        # Configure default train variational autoencoder functions
        if "train_step_fn" not in config:
            config["train_step_fn"] = jax_train_step_vae

        if "eval_step_fn" not in config:
            config["eval_step_fn"] = eval_step_vae

        super().__init__(config, model, train_ds, test_ds)

        # Define KL divergence term
        if "kl_loss_fn" not in config:
            self.kl_loss_fn = _kl_loss_fn
        else:
            self.kl_loss_fn = config["kl_loss_fn"]
        if "kl_weight" not in config:
            self.kl_weight = 0.5
        else:
            self.kl_weight = config["kl_weight"]

        # Create nnx metrics object for registering training stats
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            kl=nnx.metrics.Average("kl"),
            snr=nnx.metrics.Average("snr"),
        )

    def configure_data_iterators(
        self,
    ):
        """Configure data iterators.

        Generate training data set only from inputs.

        """
        if model.conditioner is None:
            self.dt_iterator_fn: Callable = iterate_x_dataset
            self.one_train_epoch_fn: Callable = self.one_train_epoch_x
            self.one_eval_epoch_fn: Callable = self.one_eval_epoch_x
        else:
            self.dt_iterator_fn: Callable = iterate_xy_dataset
            self.one_train_epoch_fn: Callable = self.one_train_epoch_xy
            self.one_eval_epoch_fn: Callable = self.one_eval_epoch_xy

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
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        for batch in self.dt_iterator_fn(
            self.train_ds, self.nbatches, self.batch_size, subkey1, shuffle
        ):
            # Shard data
            x = jax.device_put(batch, self.data_sharding)
            # Train
            # self.model.train() # Switch to train mode
            loss, state = self.train_step_fn(
                graphdef, state, self.criterion, self.kl_loss_fn, self.kl_weight, x, subkey2
            )
        return loss, state, key

    def one_train_epoch_xy(self, graphdef, state, key):
        """Function that defines one epoch of training for a VAE that uses input and conditioning data.

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
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        for batch in self.dt_iterator_fn(
            self.train_ds, self.nbatches, self.batch_size, subkey1, shuffle
        ):
            # Shard data
            x, y = jax.device_put(batch, self.data_sharding)
            # Train
            # self.model.train() # Switch to train mode
            loss, state = self.train_step_fn(
                graphdef, state, self.criterion, self.kl_loss_fn, self.kl_weight, x, subkey2, y
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
            key, step_key = jax.random.split(key)
            loss = self.eval_step_fn(
                self.model, self.criterion, self.kl_loss_fn, self.kl_weight, metrics, x, step_key
            )
        return loss, metrics, key

    def one_eval_epoch_xy(self, metrics, key):
        """Function that defines one epoch of evaluation for a VAE that uses input and conditioning data.

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
            x, y = jax.device_put(batch, self.data_sharding)
            # Eval step
            key, step_key = jax.random.split(key)
            loss = self.eval_step_fn(
                self.model, self.criterion, self.kl_loss_fn, self.kl_weight, metrics, x, step_key, y
            )
        return loss, metrics, key
