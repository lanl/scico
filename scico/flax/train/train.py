# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for training Flax models.

Assumes sharded batched data and data parallel training.
"""

import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import jax
import jax.numpy as jnp
from jax import lax

import optax

from flax import jax_utils
from flax.core import freeze, unfreeze
from flax.training import common_utils, train_state
from flax.traverse_util import ModelParamTraversal

try:
    from tensorflow.io import gfile  # noqa: F401
except ImportError:
    have_tf = False
else:
    have_tf = True

if have_tf:
    from flax.training import checkpoints

from scico.diagnostics import IterationStats
from scico.flax import create_input_iter
from scico.flax.train.clu_utils import get_parameter_overview
from scico.flax.train.input_pipeline import DataSetDict
from scico.metric import snr
from scico.typing import Array, Shape

ModuleDef = Any
KeyArray = Union[Array, jax._src.prng.PRNGKeyArray]
PyTree = Any
DType = Any


class ConfigDict(TypedDict):
    """Dictionary structure for training parmeters.

    Definition of the dictionary structure
    expected for specifying training parameters."""

    seed: float
    opt_type: str
    momentum: float
    batch_size: int
    num_epochs: int
    base_learning_rate: float
    lr_decay_rate: float
    warmup_epochs: int
    steps_per_eval: int
    log_every_steps: int
    steps_per_epoch: int
    steps_per_checkpoint: int
    log: bool
    workdir: str
    checkpointing: bool
    return_state: bool
    lr_schedule: Callable
    criterion: Callable
    create_train_state: Callable
    train_step_fn: Callable
    eval_step_fn: Callable
    post_lst: List[Callable]


class ModelVarDict(TypedDict):
    """Dictionary structure for Flax variables.

    Definition of the dictionary structure
    grouping all Flax model variables."""

    params: PyTree
    batch_stats: PyTree


class MetricsDict(TypedDict, total=False):
    """Dictionary structure for training metrics.

    Definition of the dictionary structure
    for metrics computed or updates made during
    training."""

    loss: float
    snr: float
    learning_rate: float


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


def compute_metrics(output: Array, labels: Array, criterion: Callable = mse_loss) -> MetricsDict:
    """Compute diagnostic metrics. Assummes sharded batched
    data (i.e. it only works inside pmap because it needs an
    axis name).

    Args:
        output: Comparison signal.
        labels: Reference signal.
        criterion: Loss function. Default: :meth:`mse_loss`.

    Returns:
        Loss and SNR between `output` and `labels`.
    """
    loss = criterion(output, labels)
    snr_ = snr(labels, output)
    metrics: MetricsDict = {
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
        config: Dictionary of configuration. The values to use correspond to `base_learning_rate`,
            `num_epochs`, `steps_per_epochs` and `lr_decay_rate`.

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


def initialize(key: KeyArray, model: ModuleDef, ishape: Shape) -> Tuple[PyTree, ...]:
    """Initialize Flax model.

    Args:
        key: A PRNGKey used as the random key.
        model: Flax model to train.
        ishape: Shape of signal (image) to process by `model`. Make sure that no batch dimension is included.

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


def create_basic_train_state(
    key: KeyArray,
    config: ConfigDict,
    model: ModuleDef,
    ishape: Shape,
    learning_rate_fn: optax._src.base.Schedule,
    variables0: Optional[ModelVarDict] = None,
) -> TrainState:
    """Create Flax basic train state and initialize.

    Args:
        key: A PRNGKey used as the random key.
        config: Dictionary of configuration. The values
           to use correspond to keywords: `opt_type`
           and `momentum`.
        model: Flax model to train.
        ishape: Shape of signal (image) to process by `model`. Make sure that no batch dimension is included.
        variables0: Optional initial state of model
           parameters. If not provided a random initialization
           is performed. Default: ``None``.
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
        raise NotImplementedError(
            f"Optimizer specified {config['opt_type']} has not been included in SCICO"
        )

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
        state: Flax train state which includes model and
            optimiser parameters.
        workdir: checkpoint file or directory of checkpoints
            to restore from.

    Returns:
        Restored `state` updated from checkpoint file,
        or if no checkpoint files present, returns the
        passed-in `state` unchanged.
    """
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state: TrainState, workdir: Union[str, os.PathLike]):  # pragma: no cover
    """Store model and optimiser state.

    Args:
        state: Flax train state which includes model and
            optimiser parameters.
        workdir: str or pathlib-like path to store checkpoint
            files in.
    """
    if jax.process_index() == 0:
        # get train state from first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


def _train_step(
    state: TrainState,
    batch: DataSetDict,
    learning_rate_fn: optax._src.base.Schedule,
    criterion: Callable,
) -> Tuple[TrainState, MetricsDict]:
    """Perform a single data parallel training step. Assumes sharded batched data.

    This function is intended to be used via :class:`BasicFlaxTrainer`, not directly.

    Args:
        state: Flax train state which includes the
           model apply function, the model parameters
           and an Optax optimizer.
        batch: Sharded and batched training data.
        learning_rate_fn: A function to map step
           counts to values.
        criterion: A function that specifies the loss being minimized in training.

    Returns:
        Updated parameters and diagnostic statistics.
    """

    def loss_fn(params: PyTree):
        """Loss function used for training."""
        output, new_model_state = state.apply_fn(
            {
                "params": params,
                "batch_stats": state.batch_stats,
            },
            batch["image"],
            mutable=["batch_stats"],
        )
        loss = criterion(output, batch["label"])
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


def clip_positive(params: PyTree, traversal: ModelParamTraversal, minval: float = 1e-4) -> PyTree:
    """Clip parameters to positive range.

    Args:
        params: Current model parameters.
        traversal: Utility to select model parameters.
        minval: Minimum value to clip selected model parameters
            and keep them in a positive range. Default: 1e-4.
    """
    params_out = traversal.update(lambda x: jnp.clip(x, a_min=minval), unfreeze(params))

    return freeze(params_out)


def clip_range(
    params: PyTree, traversal: ModelParamTraversal, minval: float = 1e-4, maxval: float = 1
) -> PyTree:
    """Clip parameters to specified range.

    Args:
        params: Current model parameters.
        traversal: Utility to select model parameters.
        minval: Minimum value to clip selected model parameters. Default: 1e-4.
        maxval: Maximum value to clip selected model parameters. Default: 1.
    """
    params_out = traversal.update(
        lambda x: jnp.clip(x, a_min=minval, a_max=maxval), unfreeze(params)
    )

    return freeze(params_out)


def _train_step_post(
    state: TrainState,
    batch: DataSetDict,
    learning_rate_fn: optax._src.base.Schedule,
    criterion: Callable,
    train_step_fn: Callable,
    post_lst: List[Callable],
) -> Tuple[TrainState, MetricsDict]:
    """Perform a single training step. A list of postprocessing
    functions (i.e. for spectral normalization or positivity
    condition, etc.) is applied after the gradient update.
    Assumes sharded batched data.

    This function is intended to be used via :class:`BasicFlaxTrainer`, not directly.

    Args:
        state: Flax train state which includes the
           model apply function, the model parameters
           and an Optax optimizer.
        batch: Sharded and batched training data.
        learning_rate_fn: A function to map step
           counts to values.
        criterion: A function that specifies the loss being minimized in training.
        train_step_fn: A function that executes a training step.
        post_lst: List of postprocessing functions to apply to parameter set after optimizer step (e.g. clip
            to a specified range, normalize, etc.).

    Returns:
        Updated parameters, fulfilling additional constraints,
        and diagnostic statistics.
    """

    new_state, metrics = train_step_fn(state, batch, learning_rate_fn, criterion)

    # Post-process parameters
    for post_fn in post_lst:
        new_params = post_fn(new_state.params)
        new_state = new_state.replace(params=new_params)

    return new_state, metrics


def _eval_step(state: TrainState, batch: DataSetDict) -> MetricsDict:
    """Evaluate current model state. Assumes sharded
    batched data.

    This function is intended to be used via :class:`BasicFlaxTrainer` or :meth:`only_evaluate`, not directly.

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
def sync_batch_stats(state: TrainState) -> TrainState:
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch
    # statistics and those are synced before evaluation
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


# pmean only works inside pmap because it needs an axis name.
#: This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


class ArgumentStruct:
    """Class that converts a python dictionary into an object with named entries given by the dictionary keys.

    After the object instantiation both modes of access (dictionary or object entries) can be used.
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)


def stats_obj() -> Tuple[IterationStats, Callable]:
    """Functionality to log and store iteration statistics.

    This function initializes an object :class:`.diagnostics.IterationStats` to log and store
    iteration statistics if logging is enabled during training.
    The statistics collected are: epoch, time, learning rate, loss and snr in training and loss and snr in evaluation.
    The :class:`.diagnostics.IterationStats` object takes care of both: printing stats to command line and storing
    them for further analysis.
    """
    # epoch, time learning rate loss and snr (train and
    # eval) fields
    itstat_fields = {
        "Epoch": "%d",
        "Time": "%8.2e",
        "Train_LR": "%.6f",
        "Train_Loss": "%.6f",
        "Train_SNR": "%.2f",
        "Eval_Loss": "%.6f",
        "Eval_SNR": "%.2f",
    }
    itstat_attrib = [
        "epoch",
        "time",
        "train_learning_rate",
        "train_loss",
        "train_snr",
        "loss",
        "snr",
    ]

    # dynamically create itstat_func; see https://stackoverflow.com/questions/24733831
    itstat_return = "return(" + ", ".join(["obj." + attr for attr in itstat_attrib]) + ")"
    scope: Dict[str, Callable] = {}
    exec("def itstat_func(obj): " + itstat_return, scope)
    default_itstat_options: Dict[str, Union[dict, Callable, bool]] = {
        "fields": itstat_fields,
        "itstat_func": scope["itstat_func"],
        "display": True,
    }
    itstat_insert_func: Callable = default_itstat_options.pop("itstat_func")  # type: ignore
    itstat_object = IterationStats(**default_itstat_options)  # type: ignore

    return itstat_object, itstat_insert_func


class BasicFlaxTrainer:
    """Class for encapsulating Flax training configuration and execution."""

    def __init__(
        self,
        config: ConfigDict,
        model: ModuleDef,
        train_ds: DataSetDict,
        test_ds: DataSetDict,
        variables0: Optional[ModelVarDict] = None,
    ):
        """Configure model training and evaluation loop.

        Assumes sharded batched data and uses data parallel training.
        Additionally, construct a Flax train state which includes the model apply function,
        the model parameters and an Optax optimizer.

        Args:
            config: Hyperparameter configuration.
            model: Flax model to train.
            train_ds: Dictionary of training data (includes images
                and labels).
            test_ds: Dictionary of testing data (includes images
                and labels).
            variables0: Optional initial state of model
                parameters. Default: ``None``.
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

    def set_training_parameters(
        self,
        config: ConfigDict,
        len_train: int,
        len_test: int,
    ):
        """Extract configuration parameters and construct training functions.

        Parameters and functions are passed in the configuration dictionary.
        Default values are used when parameters are not included in configuration.

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
            raise ValueError("Batch size must be divisible by the number of devices")
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

        - log: A flag for logging to the output terminal the evolution of results. Default: ``False``.
        - workdir: Directory to write checkpoints. Default: execution directory.
        - checkpointing: A flag for checkpointing model state.
            Default: ``False``. `RunTimeError` is generated if
            ``True`` and tensorflow is not available.
        - return_state: A flag for returning the train state instead of the model variables. Default: ``False``, i.e. return model variables.

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

        The functions constructed correspond to

        - `create_lr_schedule`: A function that creates an Optax learning rate schedule. Default:
            :meth:`create_cnst_lr_schedule`.
        - `criterion`: A function that specifies the loss being minimized in training. Default: :meth:`mse_loss`.
        - `create_train_state`: A function that creates a Flax train state and initializes it. A train state object helps to keep optimizer and module functionality grouped for training. Default:
            :meth:`create_basic_train_state`.
        - `train_step_fn`: A hook for a function that executes a training step. Default: :meth:`_train_step`, i.e. use the standard train step.
        - `eval_step_fn`: A hook for a function that executes an eval step. Default: :meth:`_eval_step`, i.e. use the standard eval step.
        - `post_lst`: List of postprocessing functions to apply to parameter set after optimizer step (e.g. clip
            to a specified range, normalize, etc.).

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
            self.train_step_fn = _train_step

        if "eval_step_fn" in config:
            self.eval_step_fn: Callable = config["eval_step_fn"]
        else:
            self.eval_step_fn = _eval_step

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
            "Channels: %d, training signals: %d, testing"
            " signals: %d, signal size: %d"
            % (
                train_ds["label"].shape[-1],
                train_ds["label"].shape[0],
                test_ds["label"].shape[0],
                train_ds["label"].shape[1],
            )
        )

    def define_parallel_training_functions(self):
        """Construct parallel versions of training functions via `jax.pmap`."""
        if self.post_lst is not None:
            self.p_train_step = jax.pmap(
                functools.partial(
                    _train_step_post,
                    train_step_fn=self.train_step_fn,
                    learning_rate_fn=self.lr_schedule,
                    criterion=self.criterion,
                    post_lst=self.post_lst,
                ),
                axis_name="batch",
            )
        else:
            self.p_train_step = jax.pmap(
                functools.partial(
                    self.train_step_fn, learning_rate_fn=self.lr_schedule, criterion=self.criterion
                ),
                axis_name="batch",
            )
        self.p_eval_step = jax.pmap(
            functools.partial(self.eval_step_fn, criterion=self.criterion),
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

        A train state object helps to keep optimizer and module functionality grouped for training.

        Args:
            config: Hyperparameter configuration.
            key: A PRNGKey used as the random key.
            model: Flax model to train.
            variables0: Optional initial state of model
                parameters. Default: ``None``.
        """
        # Create Flax training state
        state = self.create_train_state(
            key, config, model, self.ishape, self.lr_schedule, variables0
        )
        if self.checkpointing and variables0 is None:
            # Only restore if no initialization is provided
            if have_tf:  # Flax checkpointing requires tensorflow
                state = restore_checkpoint(state, self.workdir)
            else:
                raise RuntimeError(
                    "Tensorflow not available and it is required for Flax checkpointing."
                )
        self.log(get_parameter_overview(state.params))
        self.log(get_parameter_overview(state.batch_stats))

        self.state = state

    def train(self):
        """Execute training loop.

        Returns:
            Model variables extracted from TrainState  and iteration stats object obtained after executing the training loop.
            Alternatively the TrainState can be returned directly instead of the model variables.
            Note that the iteration stats object is not None only if log is enabled when configuring the training loop.
        """
        state = self.state
        step_offset = int(state.step)  # > 0 if restarting from checkpoint

        # For parallel training
        state = jax_utils.replicate(state)
        # Execute training loop and register stats
        t0 = time.time()
        self.log("Initial compilation, this might take some minutes...")

        for step, batch in zip(range(step_offset, self.num_steps), self.train_dt_iter):
            state, metrics = self.p_train_step(state, batch)
            if step == step_offset:
                self.log("Initial compilation completed.")
            if (step + 1) % self.log_every_steps == 0:
                # sync batch statistics across replicas
                state = sync_batch_stats(state)
                self.update_metrics(state, step, metrics, t0)
            if (step + 1) % self.steps_per_checkpoint == 0 or step + 1 == self.num_steps:
                # sync batch statistics across replicas
                state = sync_batch_stats(state)
                self.checkpoint(state)

        # Wait for finishing asynchronous execution
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        # Close object for iteration stats if logging
        if self.logflag:
            self.itstat_object.end()

        state = sync_batch_stats(state)
        # Extract one copy of state
        state = jax_utils.unreplicate(state)
        if self.return_state:
            return state, self.itstat_object  # type: ignore

        dvar: ModelVarDict = {
            "params": state.params,
            "batch_stats": state.batch_stats,
        }

        return dvar, self.itstat_object  # type: ignore

    def update_metrics(self, state: TrainState, step: int, metrics: MetricsDict, t0):
        """Compute metrics for current model state.

        Metrics for training and testing (eval) sets are computed and stored in an
        iteration stats object. This is executed only if logging is enabled.

        Args:
            state: Flax train state which includes the
                model apply function and the model parameters.
            step: Current step in training.
            metrics: Current diagnostic statistics computed from training set.
            t0: Time when training loop started.
        """
        if not self.logflag:
            return

        train_metrics: List[Any] = []
        eval_metrics: List[Any] = []

        # Training metrics computed in step
        train_metrics.append(metrics)
        # Build summary dictionary for logging
        # Include training stats
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f"train_{k}": v for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
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
        summary_eval = jax.tree_map(lambda x: x.mean(), eval_metrics)
        summary.update(summary_eval)

        # Update iteration stats object
        assert isinstance(self.itstat_object, IterationStats)  # for mypy
        self.itstat_object.insert(self.itstat_insert_func(ArgumentStruct(**summary)))

    def checkpoint(self, state: TrainState):  # pragma: no cover
        """Checkpoint training state if enabled (and Tensorflow available).

        Args:
            state: Flax train state.
        """
        if self.checkpointing:
            if not have_tf:  # Flax checkpointing requires tensorflow
                raise RuntimeError(
                    "Tensorflow not available and it is" " required for Flax checkpointing."
                )
            save_checkpoint(state, self.workdir)

    def log(self, logstr: str):
        """Print stats to output terminal if logging is enabled.

        Args:
            logstr: String to be logged.
        """
        if self.logflag:
            print(logstr)


def _apply_fn(model: ModuleDef, variables: ModelVarDict, batch: DataSetDict) -> Array:
    """Apply current model. Assumes sharded
    batched data and replicated variables for distributed processing.

    This function is intended to be used via :meth:`only_apply`, not directly.

    Args:
        model: Flax model to apply.
        variables: State of model parameters (replicated).
        batch: Sharded and batched training data.

    Returns:
        Output computed by given model.
    """
    output = model.apply(variables, batch["image"], train=False, mutable=False)
    return output


def only_apply(
    config: ConfigDict,
    workdir: str,
    model: ModuleDef,
    test_ds: DataSetDict,
    apply_fn: Callable = _apply_fn,
    variables: Optional[ModelVarDict] = None,
    checkpointing: bool = False,
) -> Tuple[Array, ModelVarDict]:
    """Execute model application loop.

    Args:
        config: Hyperparameter configuration.
        workdir: Directory to read checkpoint (if enabled).
        model: Flax model to apply.
        test_ds: Dictionary of testing data (includes images
            and labels).
        apply_fn: A hook for a function that applies current model. Default: :meth:`_apply_fn`, i.e. use the standard apply function.
        variables: Model parameters to use for evaluation.
            Default: ``None`` (i.e. read from checkpoint).
        checkpointing: A flag for checkpointing model state.
            Default: ``False``. `RunTimeError` is generated if
            ``True`` and tensorflow is not available.

    Returns:
        Output of model evaluated at the input provided in `test_ds`.

    Raises:
        Error if no variables and no checkpoint are specified.
    """
    if variables is None:
        if checkpointing:
            if not have_tf:
                raise RuntimeError(
                    "Tensorflow not available and it is " "required for Flax checkpointing."
                )
            state = checkpoints.restore_checkpoint(workdir, model)
            variables = {
                "params": state["params"],
                "batch_stats": state["batch_stats"],
            }
            print(get_parameter_overview(variables["params"]))
            print(get_parameter_overview(variables["batch_stats"]))
        else:
            raise Exception("No variables or checkpoint provided")

    # For distributed testing
    local_batch_size = config["batch_size"] // jax.process_count()
    size_device_prefetch = 2  # Set for GPU
    # Configure seed.
    key = jax.random.PRNGKey(config["seed"])
    # Set data iterator
    eval_dt_iter = create_input_iter(
        key,  # eval: no permutation
        test_ds,
        local_batch_size,
        size_device_prefetch,
        model.dtype,
        train=False,
    )
    p_apply_step = jax.pmap(apply_fn, axis_name="batch", static_broadcasted_argnums=0)

    # Evaluate model with provided variables
    variables = jax_utils.replicate(variables)
    num_examples = test_ds["image"].shape[0]
    steps_ = num_examples // config["batch_size"]
    output_lst = []
    for _ in range(steps_):
        eval_batch = next(eval_dt_iter)
        output_batch = p_apply_step(model, variables, eval_batch)
        output_lst.append(output_batch.reshape((-1,) + output_batch.shape[-3:]))

    # Allow for completing the async run
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # Extract one copy of variables
    variables = jax_utils.unreplicate(variables)
    # Convert to array
    output = jnp.array(output_lst)
    # Remove leading dimension
    output = output.reshape((-1,) + output.shape[-3:])

    return output, variables  # type: ignore
