import os

# Set an appropriate processor count (only applies if GPU is not available).
from multiprocessing import cpu_count

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count() // 2}"

from functools import partial
from time import time
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from optax import huber_loss

from scico import flax as sflax
from scico.flax.examples import load_image_data
from scico.flax_nnx.diffusion.diagnostics import stats_obj
from scico.flax_nnx.diffusion.models import ConditionalUNet
from scico.flax_nnx.diffusion.trainer_input_transform import FlaxNNXScoreTrainer
from scico.flax_nnx.utils import save_model

"""
Read data from cache or generate if not available.
"""
size = 64  # patch size
train_nimg = 400  # number of training images
test_nimg = 20  # number of testing images
nimg = train_nimg + test_nimg
gray = True  # use gray scale images
data_mode = "dn"  # Denoising problem
noise_level = 0.0  # Standard deviation of noise (noisy images not used)
noise_range = False  # Use fixed noise level
stride = 23  # Stride to sample multiple patches from each image

train_ds, test_ds = load_image_data(
    train_nimg,
    test_nimg,
    size,
    gray,
    data_mode,
    verbose=True,
    noise_level=noise_level,
    noise_range=noise_range,
    stride=stride,
)


"""
Define training configuration dictionary.
"""
train_conf: sflax.ConfigDict = {
    "seed": 12345,
    "opt_type": "ADAM",
    "batch_size": 16,
    "num_epochs": 50,
    "base_learning_rate": 1e-4,
    "warmup_epochs": 0,
    "log_every_epochs": 5,
    "log": True,
    "checkpointing": True,
}


"""
Construct conditional UNet model.
"""
model = ConditionalUNet(
    shape=(size, size),
    channels=1,
    init_channels=size,
    dim_mults=(
        1,
        2,
        4,
    ),
)


"""
Set up noise levels for training.
"""
σ0 = 5e-2
Nσlev = 8
σv = jnp.array([(k > 0) * σ0 * 2 ** (k - 1) for k in range(Nσlev)])


"""
Override the trainer step t computation so that t represents noise
standard deviation instead of time.
"""


def _step_t(batch_x: ArrayLike, key: ArrayLike, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    sigvec = kwargs.pop("sigvec", jnp.array([0.0]))
    tshp = (batch_x.shape[0],) + (1,) * len(batch_x.shape[1:])
    rndidx = jax.random.randint(key, (batch_x.shape[0], 1), minval=0, maxval=sigvec.size)
    batch_t = sigvec[rndidx]
    t = batch_t.reshape(tshp)
    return t, batch_t


"""
Override the trainer step x computation so that the batch noise vector is
just the noise standard devitation vector rather than depending on a time
schedule.
"""


def _step_x(
    batch: ArrayLike, key: ArrayLike, t: ArrayLike, **kwargs
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    z = jax.random.normal(key, batch.shape)
    std = t
    batch_x = batch + z * std
    return z, std, batch_x


"""
Override the trainer loss computation so that the noise standard
deviation multiplies the noise rather than the CNN output.
"""


def _step_loss(criterion: Callable, z: ArrayLike, std: ArrayLike, output: ArrayLike) -> ArrayLike:
    return criterion(output, -z * std)


"""
Configure trainer.
"""
workdir = os.path.join("/tmp", "scico", "examples", "cundn_out")
train_conf["workdir"] = workdir
train_conf["criterion"] = huber_loss
train_conf["step_t"] = partial(_step_t, sigvec=σv)
train_conf["step_x"] = _step_x
train_conf["step_loss"] = _step_loss
train_conf["stats_obj"] = stats_obj()


"""
Construct training object and train.
"""
trainer = FlaxNNXScoreTrainer(
    train_conf,
    model,
    train_ds,
    test_ds,
)

key = jax.random.PRNGKey(0x1234)
start_time = time()
stats_object = trainer.train(key)
time_train = time() - start_time


"""
Save trained model. Model can be loaded using the `load_model` function.
"""
save_model(model, os.path.join(workdir, "cun_bsds_model.pkl"))
