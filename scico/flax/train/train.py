#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training a Flax model using SPMD
(Adapted from imagenet example from Flax github repo.)
"""
import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import parameter_overview

import ml_collections

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

import flax
from flax import jax_utils
from flax.core import freeze, unfreeze

from flax.training import train_state
from flax.training import checkpoints
from flax.training import common_utils

import optax

import networks.models as mdl
import utils.external_spectral_norm as norm


def create_model(*, model_cls, **kwargs):
    return model_cls(**kwargs)


def initialized(key, image_size, size_device_prefetch, model):
    input_shape = (size_device_prefetch, image_size, image_size, model.channels)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({"params": key}, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, size_device_prefetch, learning_rate_fn
):
    """Create initial training state."""

    params, batch_stats = initialized(rng, image_size, size_device_prefetch, model)
    logging.info(parameter_overview.get_parameter_overview(params))

    if config.opt_type == "SGD":
        tx = optax.sgd(
            learning_rate=learning_rate_fn, momentum=config.momentum, nesterov=True
        )
    elif config.opt_type == "ADAM":
        tx = optax.adam(
            learning_rate=learning_rate_fn,
        )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )

    return state


def train_step(state, batch, learning_rate_fn):
    """Perform a single training step."""
    def loss_fn(params):
        """Loss function used for training."""
        output, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats,},
            batch['image'], mutable=['batch_stats'])
        loss = mse_loss(output, batch['label'])
        return loss, (new_model_state, output)

    step = state.step
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    #Re-use same axis_name as in call to pmap
    grads = lax.pmean(grads, axis_name='batch')
    new_model_state, output = aux[1]
    metrics = compute_metrics(output, batch['label'])
    metrics['learning_rate'] = lr

    # Update params and stats
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
    )

    return new_state, metrics


def train_step_extsn(state, batch, learning_rate_fn):
    """Perform a single training step."""
    def loss_fn(params):
        """Loss function used for training."""
        output, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats,},
            batch['image'], mutable=['batch_stats'])
        loss = mse_loss(output, batch['label'])
        return loss, (new_model_state, output)

    step = state.step
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    #Re-use same axis_name as in call to pmap
    grads = lax.pmean(grads, axis_name='batch')
    new_model_state, output = aux[1]
    metrics = compute_metrics(output, batch['label'])
    metrics['learning_rate'] = lr

    # Update params and stats
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
    )
    # Apply spectral normalization
    new_params = compute_spectral_normalization(new_state.params, batch['label'][0].shape)
    new_state = new_state.replace(params=new_params)

    return new_state, metrics

def eval_step(state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats,}
    output = state.apply_fn(
        variables, batch['image'], train=False, mutable=False)
    return compute_metrics(output, batch['label'])



def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each devide has its own version of the running average batch
    # statistics and those are synced before evaluation
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))



def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, train_ds, test_ds) -> TrainState:
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration.
      workdir: Directory to write tensorboard summaries.

    Returns:
      Final TrainState.
    """
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    rng = random.PRNGKey(config.seed)
    key1, key2, key3 = random.split(rng, 3)

    image_size = config.output_size

    if config.batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    local_batch_size = config.batch_size // jax.process_count()
    if jax.device_count() > 0:
        size_device_prefetch = config.batch_size // jax.device_count()
    else:
        size_device_prefetch = 2

    input_dtype = jnp.float32

    train_iter = create_input_iter(key1,
        train_ds,
        local_batch_size,
        size_device_prefetch,
        input_dtype,
        train=True,
        cache=config.cache,
    )

    eval_iter = create_input_iter(key2,
        test_ds,
        local_batch_size,
        size_device_prefetch,
        input_dtype,
        train=False,
        cache=config.cache,
    )

    steps_per_epoch = ( train_ds['image'].shape[0] // config.batch_size )

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    num_validation_examples = test_ds['image'].shape[0]
    if config.steps_per_eval == -1:
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 10

    #base_learning_rate = config.learning_rate * config.batch_size / 256.0
    base_learning_rate = config.learning_rate# * config.batch_size / 256.0

    channels = train_ds['image'].shape[-1]
    numtrain = train_ds['image'].shape[0]
    logging.info('Channels: %d, training images: %d, training patches: %d, testing patches: %d, patch size: %d', channels, config.num_img, numtrain, num_validation_examples, train_ds['image'].shape[1])
    model_cls = getattr(mdl, config.model)
    model = create_model(model_cls=model_cls,
                         depth=config.depth,
                         channels=channels,
                         num_filters=config.num_filters,
                         block_depth=config.block_depth,
                         dtype=input_dtype)

    learning_rate_fn = create_learning_rate_fn(
        config, base_learning_rate, steps_per_epoch
    )

    state = create_train_state(key3, config, model, image_size, size_device_prefetch, learning_rate_fn)
    state = restore_checkpoint(state, workdir)
    step_offset = int(state.step) # > 0 if restarting from checkpoint
    state = jax_utils.replicate(state)

    if config.extsn:
        p_train_step = jax.pmap(
            functools.partial(train_step_extsn, learning_rate_fn=learning_rate_fn),
            axis_name='batch')
    else:
        p_train_step = jax.pmap(
            functools.partial(train_step, learning_rate_fn=learning_rate_fn),
            axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')

    train_metrics = []
    #hooks = []
    #if jax.process_index() == 0:
    #    hooks += [periodic_actions.Profile(num_profile_steps=5,
    #                                       logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in zip(range(step_offset, num_steps), train_iter):
        state, metrics = p_train_step(state, batch)
        #for h in hooks:
        #    h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')

        if config.get('log_every_steps'):
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f'train_{k}':v
                    for k, v in jax.tree_map(lambda x: x.mean(),
                                             train_metrics).items()
                }
                summary['steps_per_second'] = config.log_every_steps / (
                    time.time() - train_metrics_last_t)
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []

            # sync batch statistics across replicas
            state = sync_batch_stats(state)
            for _ in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info('eval epoch: %d, loss: %.4f, psnr: %.2f',
                         epoch, summary['loss'], summary['psnr'])
            writer.write_scalars(
                step + 1, {f'eval_{key}': val for key, val in summary.items()})
            writer.flush()
        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            state = sync_batch_stats(state)
            save_checkpoint(state, workdir)

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state


def only_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    """Execute evaluation.
    Args:
      config: Hyperparameter configuration.
      workdir: Directory to write tensorboard summaries.

    Returns:
      State read from checkpoint.
    """

    _, test_ds = build_img_dataset(config)
    channels = test_ds['image'].shape[-1]
    logging.info('Channels: %d, testing patches: %d, patch size: %d', channels, test_ds['image'].shape[0], test_ds['image'].shape[1])

    input_dtype = jnp.float32
    model_cls = getattr(mdl, config.model)
    model = create_model(model_cls=model_cls,
                         depth=config.depth,
                         channels=channels,
                         num_filters=config.num_filters,
                         block_depth=config.block_depth,
                         dtype=input_dtype)

    state = checkpoints.restore_checkpoint(workdir, model)

    # Eval all
    variables = {'params': state['params'], 'batch_stats': state['batch_stats'],}
    output = model.apply(
        variables, test_ds['image'], train=False, mutable=False)

    import numpy as np

    np.save('infer_test_in', jax.device_get(test_ds['image']))
    np.save('infer_test_out', jax.device_get(test_ds['label']))
    np.save('infer_test_pred', jax.device_get(output))

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state

