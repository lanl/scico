# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for training Flax models.

Assummes sharded batched data and data parallel training.
"""

import functools
import os
import time
from typing import Any, Callable, List, Optional, TypedDict, Union

import jax
import jax.numpy as jnp
from jax import lax

import optax

from flax import jax_utils
from flax.core import freeze, unfreeze
from flax.training import common_utils, train_state
from flax.traverse_util import ModelParamTraversal

try:
    import clu
except ImportError:
    have_clu = False
else:
    have_clu = True

try:
    from tensorflow.io import gfile
except ImportError:
    have_tf = False
else:
    have_tf = True

if have_tf:
    from flax.training import checkpoints

from scico.flax import create_input_iter
from scico.flax.train.input_pipeline import DataSetDict
from scico.metric import snr
from scico.typing import Array, Shape

ModuleDef = Any
KeyArray = Union[Array, jax._src.prng.PRNGKeyArray]
PyTree = Any


class ConfigDict(TypedDict):
    """Definition of the dictionary structure
    expected for the data sets for training."""

    seed: float
    depth: int
    num_filters: int
    block_depth: int
    opt_type: str
    momentum: float
    batch_size: int
    num_epochs: int
    base_learning_rate: float
    lr_decay_rate: float
    warmup_epochs: int
    num_train_steps: int
    steps_per_eval: int
    log_every_steps: int
    steps_per_epoch: int


class ModelVarDict(TypedDict):
    """Definition of the dictionary structure
    for including all Flax model variables."""

    params: PyTree
    batch_stats: PyTree


# Loss Function
def mse_loss(output: Array, labels: Array) -> float:
    """
    Compute Mean Squared Error (MSE) loss for training
    via Optax.

    Args:
        output: Comparison signal.
        labels: Reference signal.

    Returns:
        MSE between `output` and `labels`.
    """
    mse = optax.l2_loss(output, labels)
    return jnp.mean(mse)


def compute_metrics(output: Array, labels: Array):
    """Compute diagnostic metrics. Assummes sharded batched data (i.e. it only works inside pmap because it needs an axis name).

    Args:
        output: Comparison signal.
        labels: Reference signal.

    Returns:
        MSE and SNR between `output` and `labels`.
    """
    loss = mse_loss(output, labels)
    snr_ = snr(labels, output)
    metrics = {
        "loss": loss,
        "snr": snr_,
    }
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


# Learning rate
def create_cnst_lr_schedule(config: ConfigDict) -> optax._src.base.Schedule:
    """Create learning rate to be a constant specified
    value.

    Args:
        config: Dictionary of configuration. The value
           to use corresponds to the `base_learning_rate`
           keyword.

    Returns:
        schedule: A function that maps step counts to values.
    """
    schedule = optax.constant_schedule(config["base_learning_rate"])
    return schedule


def create_exp_lr_schedule(config: ConfigDict) -> optax._src.base.Schedule:
    """Create learning rate schedule to have an exponential decay.

    Args:
        config: Dictionary of configuration. The values
           to use corresponds to the `base_learning_rate`
           , `num_epochs`, `steps_per_epochs` and `lr_decay_rate`.

    Returns:
        schedule: A function that maps step counts to values.
    """
    decay_steps = config["num_epochs"] * config["steps_per_epoch"]
    schedule = optax.exponential_decay(
        config["base_learning_rate"], decay_steps, config["lr_decay_rate"]
    )
    return schedule


def create_cosine_lr_schedule(config: ConfigDict) -> optax._src.base.Schedule:
    """Create learning rate to follow a pre-specified
    schedule with warmup and cosine stages.

    Args:
        config: Dictionary of configuration. The parameters
           to use correspond to keywords: `base_learning_rate`,
           `num_epochs`, `warmup_epochs` and `steps_per_epoch`.

    Returns:
        schedule: A function that maps step counts to values.
    """
    # Warmup stage
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config["base_learning_rate"],
        transition_steps=config["warmup_epochs"] * config["steps_per_epoch"],
    )
    # Cosine stage
    cosine_epochs = max(config["num_epochs"] - config["warmup_epochs"], 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config["base_learning_rate"],
        decay_steps=cosine_epochs * config["steps_per_epoch"],
    )

    schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config["warmup_epochs"] * config["steps_per_epoch"]],
    )

    return schedule


def initialize(key: KeyArray, model: ModuleDef, ishape: Shape):
    """Initialize Flax model.

    Args:
        key: A PRNGKey used as the random key.
        model: Flax model to train.
        ishape: Shape of signal (image) to process by `model`.

    Returns:
        Initial model parameters (including `batch_stats`).
    """
    input_shape = (1, ishape[0], ishape[1], model.channels)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({"params": key}, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]


# Flax Train State
class TrainState(train_state.TrainState):
    """Definition of Flax train state including
    `batch_stats` for batch normalization."""

    batch_stats: Any


def create_train_state(
    key: KeyArray,
    config: ConfigDict,
    model: ModuleDef,
    ishape: Shape,
    learning_rate_fn: optax._src.base.Schedule,
    variables0: Optional[ModelVarDict] = None,
) -> TrainState:
    """Create initial training state.

    Args:
        key: A PRNGKey used as the random key.
        config: Dictionary of configuration. The values
           to use correspond to keywords: `opt_type`
           and `momentum`.
        model: Flax model to train.
        ishape: Shape of signal (image) to process by `model`.
        variables0: Optional initial state of model parameters. If not provided a random initialization is performed. Default: ``None``.
        learning_rate_fn: A function that maps step
           counts to values.

    Returns:
        state: Flax train state which includes the
           model apply function, the model parameters
           and an Optax optimizer.
    """
    if variables0 is None:
        params, batch_stats = initialize(key, model, ishape)
    else:
        params = variables0["params"]
        batch_stats = variables0["batch_stats"]

    if config["opt_type"] == "SGD":
        # Stochastic Gradient Descent optimiser
        tx = optax.sgd(learning_rate=learning_rate_fn, momentum=config["momentum"], nesterov=True)
    elif config["opt_type"] == "ADAM":
        # Adam optimiser
        tx = optax.adam(
            learning_rate=learning_rate_fn,
        )
    elif config["opt_type"] == "ADAMW":
        # Adam with weight decay regularization
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
        )
    else:
        raise NotImplementedError

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )

    return state


# Flax checkpoints
def restore_checkpoint(
    state: TrainState, workdir: Union[str, os.PathLike]
) -> TrainState:  # pragma: no cover
    """Load model and optimiser state.

    Args:
        state: Flax train state which includes model and optimiser parameters.
        workdir: checkpoint file or directory of checkpoints to restore from.

    Returns:
        Restored `state` updated from checkpoint file,
        or if no checkpoint files present, returns the
        passed-in `state` unchanged.
    """
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state: TrainState, workdir: Union[str, os.PathLike]):  # pragma: no cover
    """Store model and optimiser state.

    Args:
        state: Flax train state which includes model and optimiser parameters.
        workdir: str or pathlib-like path to store checkpoint files in.
    """
    if jax.process_index() == 0:
        # get train state from first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


def train_step(state: TrainState, batch: DataSetDict, learning_rate_fn: optax._src.base.Schedule):
    """Perform a single training step. Assummes sharded batched data.

    Args:
        state: Flax train state which includes the
           model apply function, the model parameters
           and an Optax optimizer.
        batch: Sharded and batched training data.
        learning_rate_fn: A function that maps step
           counts to values.

    Returns:
        Updated parameters and diagnostic statistics.
    """

    def loss_fn(params):
        """Loss function used for training."""
        output, new_model_state = state.apply_fn(
            {
                "params": params,
                "batch_stats": state.batch_stats,
            },
            batch["image"],
            mutable=["batch_stats"],
        )
        loss = mse_loss(output, batch["label"])
        return loss, (new_model_state, output)

    step = state.step
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in call to pmap
    grads = lax.pmean(grads, axis_name="batch")
    new_model_state, output = aux[1]
    metrics = compute_metrics(output, batch["label"])
    metrics["learning_rate"] = lr

    # Update params and stats
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state["batch_stats"],
    )

    return new_state, metrics


def construct_traversal(prmname: str) -> ModelParamTraversal:
    """Construct utility to select model parameters using a name filter.

    Args:
        prmname: Name of parameter to select.

    Returns:
        Flax utility to traverse and select model parameters.
    """
    return ModelParamTraversal(lambda path, _: prmname in path)


def clip_positive(params: PyTree, traversal: ModelParamTraversal, minval: float = 1e-4):
    """Clip parameters to positive range.

    Args:
        params: Current model parameters.
        traversal: Utility to select model parameters.
        minval: Minimum value to clip parameters and keep them in a positive range. Default: 1e-4.
    """
    params_out = traversal.update(lambda x: jnp.clip(x, a_min=minval), unfreeze(params))

    return freeze(params_out)


def train_step_post(
    state: TrainState,
    batch: DataSetDict,
    learning_rate_fn: optax._src.base.Schedule,
    post_fn: Callable,
):
    """Perform a single training step. A postprocessing function (i.e. for spectral normalization or positivity condition, etc.) is applied after the gradient update. Assummes sharded batched data.

    Args:
        state: Flax train state which includes the
           model apply function, the model parameters
           and an Optax optimizer.
        batch: Sharded and batched training data.
        learning_rate_fn: A function that maps step
           counts to values.
        post_fn: A postprocessing function for clipping parameter range or normalizing parameter.

    Returns:
        Updated parameters, fulfilling additional constraints, and diagnostic statistics.
    """

    new_state, metrics = train_step(state, batch, learning_rate_fn)

    # Post-process parameters
    new_params = post_fn(new_state.params)
    new_state = new_state.replace(params=new_params)

    return new_state, metrics


def eval_step(state: TrainState, batch: DataSetDict):
    """Evaluate current model state. Assummes sharded
    batched data.

    Args:
        state: Flax train state which includes the
           model apply function and the model parameters.
        batch: Sharded and batched training data.

    Returns:
        Current diagnostic statistics.
    """
    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
    }
    output = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    return compute_metrics(output, batch["label"])


# sync across replicas
def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch
    # statistics and those are synced before evaluation
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


# pmean only works inside pmap because it needs an axis name.
#: This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


def train_and_evaluate(
    config: ConfigDict,
    workdir: str,
    model: ModuleDef,
    train_ds: DataSetDict,
    test_ds: DataSetDict,
    create_lr_schedule: Callable = create_cnst_lr_schedule,
    training_step_fn: Callable = train_step,
    variables0: Optional[ModelVarDict] = None,
    checkpointing: bool = False,
    log: bool = False,
) -> ModelVarDict:
    """Execute model training and evaluation loop.

    Args:
        config: Hyperparameter configuration.
        workdir: Directory to write checkpoints and tensorboard summaries (the latter only if `clu` is available).
        model: Flax model to train.
        train_ds: Dictionary of training data (includes images and labels).
        test_ds: Dictionary of testing data (includes images and labels).
        create_lr_schedule: A function that creates an Optax learning rate schedule. Default: :meth:`create_cnst_schedule`.
        training_step_fn: A function that executes a training step. Default: :meth:`training_step`.
        variables0: Optional initial state of model parameters. Default: ``None``.
        checkpointing: A flag for checkpointing model state. Default: ``False``. `RunTimeError` is generated if ``True`` and tensorflow is not available.
        log: A flag for logging. If `clu` is available a tensorboard summary is also generated during logging. Default: ``False``.

    Returns:
        Model variables extracted from TrainState.
    """
    if log:  # pragma: no cover
        print(
            "Channels: %d, training signals: %d, testing signals: %d, signal size: %d"
            % (
                train_ds["label"].shape[-1],
                train_ds["label"].shape[0],
                test_ds["label"].shape[0],
                train_ds["label"].shape[1],
            )
        )
        if have_clu:
            from clu import metric_writers

            writer = metric_writers.create_default_writer(
                logdir=workdir, just_logging=jax.process_index() != 0
            )

    # Configure seed.
    key = jax.random.PRNGKey(config["seed"])
    # Split seed for data iterators and model initialization
    key1, key2 = jax.random.split(key)

    # Determine sharded vs. batch partition
    if config["batch_size"] % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    local_batch_size = config["batch_size"] // jax.process_count()
    size_device_prefetch = 2  # Set for GPU

    # Determine monitoring steps
    steps_per_epoch = train_ds["image"].shape[0] // config["batch_size"]
    config["steps_per_epoch"] = steps_per_epoch  # needed for creating lr schedule
    if config["num_train_steps"] == -1:
        num_steps = int(steps_per_epoch * config["num_epochs"])
    else:
        num_steps = config["num_train_steps"]
    num_validation_examples = test_ds["image"].shape[0]
    if config["steps_per_eval"] == -1:
        steps_per_eval = num_validation_examples // config["batch_size"]
    else:
        steps_per_eval = config["steps_per_eval"]
    steps_per_checkpoint = steps_per_epoch * 10

    # Construct data iterators
    train_dt_iter = create_input_iter(
        key1,
        train_ds,
        local_batch_size,
        size_device_prefetch,
        model.dtype,
        train=True,
    )
    eval_dt_iter = create_input_iter(
        key1,  # eval: no permutation
        test_ds,
        local_batch_size,
        size_device_prefetch,
        model.dtype,
        train=False,
    )

    # Create Flax training state
    ishape = train_ds["image"].shape[1:3]
    lr_schedule = create_lr_schedule(config)
    state = create_train_state(key2, config, model, ishape, lr_schedule, variables0)
    if checkpointing and variables0 is None:
        # Only restore if no initialization is provided
        if have_tf:  # Flax checkpointing requires tensorflow
            state = restore_checkpoint(state, workdir)
        else:
            raise RuntimeError(
                "Tensorflow not available and it is required for Flax checkpointing."
            )
    if log and have_clu:  # pragma: no cover
        from clu import parameter_overview

        print(parameter_overview.get_parameter_overview(state.params))
        print(parameter_overview.get_parameter_overview(state.batch_stats))
    step_offset = int(state.step)  # > 0 if restarting from checkpoint

    # For parallel training
    state = jax_utils.replicate(state)
    p_train_step = jax.pmap(
        functools.partial(training_step_fn, learning_rate_fn=lr_schedule), axis_name="batch"
    )
    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    # Execute training loop and register stats
    train_metrics: List[Any] = []
    train_metrics_last_t = time.time()
    if log:
        print("Initial compilation, this might take some minutes...")

    for step, batch in zip(range(step_offset, num_steps), train_dt_iter):
        state, metrics = p_train_step(state, batch)
        if log and step == step_offset:
            print("Initial compilation completed.")

        if log:  # pragma: no cover
            train_metrics.append(metrics)
            if (step + 1) % config["log_every_steps"] == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f"train_{k}": v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                summary["steps_per_second"] = config["log_every_steps"] / (
                    time.time() - train_metrics_last_t
                )
                print(
                    "step: %d, steps_per_second: %.6f, train_learning_rate: %.6f, train_loss: %.6f, train_snr: %.2f"
                    % (
                        step,
                        summary["steps_per_second"],
                        summary["train_learning_rate"],
                        summary["train_loss"],
                        summary["train_snr"],
                    )
                )
                if have_clu:
                    writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []

            # sync batch statistics across replicas
            state = sync_batch_stats(state)
            for _ in range(steps_per_eval):
                eval_batch = next(eval_dt_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            if log:  # pragma: no cover
                summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
                print(
                    "eval epoch: %d, loss: %.6f, snr: %.2f"
                    % (epoch, summary["loss"], summary["snr"])
                )
                if have_clu:
                    writer.write_scalars(
                        step + 1, {f"eval_{key}": val for key, val in summary.items()}
                    )
                    writer.flush()
        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            state = sync_batch_stats(state)
            if checkpointing:  # pragma: no cover
                if not have_tf:  # Flax checkpointing requires tensorflow
                    raise RuntimeError(
                        "Tensorflow not available and it is required for Flax checkpointing."
                    )
                save_checkpoint(state, workdir)

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    state = sync_batch_stats(state)
    # Extract one copy of state
    state = jax_utils.unreplicate(state)
    dvar: ModelVarDict = {
        "params": state.params,
        "batch_stats": state.batch_stats,
    }

    return dvar


def only_evaluate(
    config: ConfigDict,
    workdir: str,
    model: ModuleDef,
    test_ds: DataSetDict,
    variables: Optional[ModelVarDict] = None,
    checkpointing: bool = False,
) -> Array:
    """Execute model evaluation loop.

    Args:
        config: Hyperparameter configuration.
        workdir: Directory to read checkpoint (if enabled).
        model: Flax model to apply.
        test_ds: Dictionary of testing data (includes images and labels).
        variables: Model parameters to use for evaluation. Default: ``None`` (i.e. read from checkpoint).
        checkpointing: A flag for checkpointing model state. Default: ``False``. `RunTimeError` is generated if ``True`` and tensorflow is not available.

    Returns:
        Output of model evaluated at the input provided in `test_ds`.

    Raises:
        Error if no state and no checkpoint are specified.
    """
    if variables is None:
        if checkpointing:
            if not have_tf:
                raise RuntimeError(
                    "Tensorflow not available and it is required for Flax checkpointing."
                )
            state = checkpoints.restore_checkpoint(workdir, model)
            variables = {
                "params": state["params"],
                "batch_stats": state["batch_stats"],
            }
            if have_clu:
                from clu import parameter_overview

                print(parameter_overview.get_parameter_overview(variables["params"]))
                print(parameter_overview.get_parameter_overview(variables["batch_stats"]))
        else:
            raise Exception("No variables or checkpoint provided")

    # Evaluate model with provided variables
    output = model.apply(variables, test_ds["image"], train=False, mutable=False)

    # Allow for completing the async run
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return output, variables
