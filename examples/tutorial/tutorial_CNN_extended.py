"""
# Example: Data-Driven Priors for Inverse Problems

In this example,
 we will explore how to use CNN models as data-driven priors in a SCICO pipeline for performing computed tomography (CT) reconstruction.
"""

"""
## Setup
To set up your environment, run the cell below.

If you get a popup with 'Warning: This notebook was not authored by Google.', select 'Run anyway'.
You should see console outputs appearing.
The install may take several minutes;
when it is finished, you should see `==done with install==`.

"""
!pip install -q condacolab
import condacolab
condacolab.install()

!pip install git+https://github.com/lanl/scico@cristina/more-flax
!pip install xdesign
!conda install -c astra-toolbox astra-toolbox

print('==done with install==')

"""
## Introduction

Suppose that you are performing CT scans of many similar objects
 and want to construct a pipeline to rapidly compute the reconstruction
 of each new measurement.
For this example, we will use computer-generated foam images
as the objects we want to image.
Run the next cell to generate and visualize one of such foams.
"""

import numpy as np

import matplotlib.pyplot as plt
from xdesign import Foam, discrete_phantom

from scico import plot

plot.config_notebook_plotting()  # set up plotting
plt.rcParams["image.cmap"] = "gray"  # set default colormap

np.random.seed(7654)

N = 256  # image size
x_fm = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
x_gt = x_fm / np.max(x_fm)
x_gt = np.clip(x_gt, 0, 1.0)

# Plot signal
fig, ax = plt.subplots()
ax.imshow(x_gt)
ax.set_title("Foam")
fig.show()

"""
This image shows one foam,
 but data-driven pipelines assume that you have access to a wealth of relevant data.
 Let's generate several different foams to use in our CT reconstruction pipeline.
 Since we are interested in CT reconstruction, we need to generate both images and sinograms.

SCICO provides CT projectors based on Python libraries such as ASTRA and SVMBIR. In this case we will use the
ASTRA interface (see https://scico.readthedocs.io/en/latest/_autosummary/scico.linop.radon_astra.html).

**Define an ASTRA SCICO CT projector assuming 60 equally spaced projections.**
"""
# startq

n_projection = ...  # number of projections
angles = ...
A = ...
# starta
from scico.linop.radon_astra import TomographicProjector

n_projection = 60  # number of projections
angles = np.linspace(0, np.pi, n_projection, endpoint=False, dtype=np.float32)  # evenly spaced projection angles
A = TomographicProjector(x_gt.shape, 1, N, angles)  # Radon transform operator
# endqa

"""
Machine learning algorithms are typically very sensitive to the scaling of their inputs.
For this reason, we normalize the operator `A` by the dimension of the image,
which, for this operator, makes $ ||Ax|| \approx ||x||$.
"""

A = A / N

"""
**Test your operator by computing the sinogram of the generated foam and plotting your results.**
"""
# startq
from mpl_toolkits.axes_grid1 import make_axes_locatable

sino = ...

fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(15, 5))
plot.imview(..., title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    ...,
    title="Sinogram",
    cbar=None,
    fig=fig,
    ax=ax[1],
)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
fig.show()
# starta
from mpl_toolkits.axes_grid1 import make_axes_locatable

sino = A @ x_gt

fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(15, 5))
plot.imview(x_gt, title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    sino,
    title="Sinogram",
    cbar=None,
    fig=fig,
    ax=ax[1],
)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
fig.show()
# endqa

"""
Now, we repeat the proces to generate at least 24 different (foam, sinogram) pairs.
 **Start by generating the foams.**
You'll need to make sure that you return an `ndarray`, not a `list`.
The function `np.stack` may be useful for that.
"""
# startq
nfoams = ...
foam_collection = ...
# starta
nfoams = 24
foam_collection = np.zeros((nfoams, N, N))
for i in range(nfoams):
    print(i, end=", ")
    x_ = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
    x_ = x_ / np.max(x_)
    foam_collection[i] = np.clip(x_, 0, 1.0)
# endqa

"""
Run the next cell to plot the generated foams.
"""
nrows = 4
ncols = 6
fig, ax = plot.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
for i in range(nrows):
    for j in range(ncols):
        plot.imview(foam_collection[i * ncols + j], cbar=None, fig=fig, ax=ax[i, j])
    divider = make_axes_locatable(ax[i, j])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(ax[i, j].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

"""
Operations can be accelerated by distributing processing among GPUs or CPUs. Depending on the environment, the distribution can happen automatically (e.g. GPUs in the same node), require setting a few flags (e.g. to use multiple CPU cores), or using further packages (e.g. ray or mpi4py for multinode).

Running in Colab somewhat limits resources and exploitation of these options. Therefore, we will limit on this tutorial to run on only one CPU node, and reduce sizes of data sets, ML architectures, epochs, etc. Users are encouraged to look through the documentation and the usage examples demonstrating other distributed architectures.

For example, next cell has commented out the commands that would be useful to exploit multiple cores in a CPU. The only commands active are the ones that print the current configuration of the jax environment.
"""

#import os

import jax

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
#platform = jax.lib.xla_bridge.get_backend().platform
#print("Platform: ", platform)
print(f"{'JAX process: '}{jax.process_index()}{' / '}{jax.process_count()}")
print(f"{'JAX local devices: '}{jax.local_devices()}")

"""
For purely jax functionality, a distributed processing can be computed via `jax.vmap`. However, the CT operator uses a python (not JAX) library.
In that case we can distribute the processing via `jax.lax`. Run the next cell to distribute the computation of the sinograms. (This would be efficient in an environment with multiple resources).
"""
sino_collection = jax.lax.map(lambda x: A @ x, foam_collection)

"""
**Check the shape of the result.**
"""
# startq
...
# starta
sino_collection.shape
# endqa

"""
**Explain the shape of `sino_collection`**
"""

# startq
"""
The shape corresponds to ...
"""

# starta
"""
The shape corresponds to (number of foams, number of projections, foam dimension)
"""
# endqa

"""
You are done with part 1. Please report back in the Webex chat: **done with part 1**.

While you wait for others to finish, you could explore other SCICO linear operators that can be used to transform image data, e.g., `Convolve`.

🛑 **PAUSE HERE** 🛑
"""

"""
# Solving a regularized reconstruction: Data-Driven Priors

When there is an explicit representation of the forward model, the signal reconstruction can be posed as
a regularized least squares problem

$$ \min_\mathbf{x} \| \mathbf{y} - A \mathbf{x} \|_2^2 + \, \lambda \, r(\mathbf{x}).$$

For example, in the CT case, the forward model $A$ is the CT projector, the measurements are the sinograms $y$ and
the solution $x$ represents the signal reconstruction. The constant $\lambda > 0$, establishes the trade-off
between the fidelity to the measurements and the regularization of the solution represented as $r(x)$.

In many cases, the regularization is necessary to find a meaningful solution for ill-posed problems. The difficulty
arises in specifying an efficient regularization criterion. Functions like TV are adequate for piece-wise constant
solutions, but may be not expressive enough for more general cases. Frameworks like plug-and-play priors provide
a convenient alternative for cases when denoisers implement appropriate artifact removal. A complementary
strategy is to unroll the iterative optimization process and build a neural network model that can be trained
end-to-end. This kind of pipeline also offers the benefit of rapid evaluation in deployment, although its
performance will depend on the training data, as is usually the case in machine learning models.

The following diagram illustrates the kind of ML structure we will be training for the CT reconstruction:

![Unrolled end-to-end](../../examples/tutorial/unrolled.png "Unrolled end-to-end")

In the diagram, the green blocks correspond to a denoiser, generally a residual convolutional neural network, and are trainable. The red blocks correspond to a data consistency block and use the forward and adjoint operators. We will be constructing and training one such unrolled model.
"""

"""
For this tutorial we will use the MoDL architecture.

The class [flax.MoDLNet](../_autosummary/scico.learning.rst#scico.learning.MoDL)
 implements the MoDL architecture, which unrolls the optimization problem

  $$\mathrm{argmin}_{\mathbf{x}} \; \| A \mathbf{x} - \mathbf{y} \|_2^2 + \lambda \, \| \mathbf{x} - \mathrm{D}_w(\mathbf{x})\|_2^2 \;,$$

where $A$ is a forward operator, in this case a CT projector, $\mathbf{y}$ is a set of measurements, in this case a collection of sinograms, $\mathrm{D}_w$ is the
 regularization (a denoiser), and $\mathbf{x}$ is the set of reconstructed images.
  The MoDL abstracts the iterative solution by an unrolled network where each iteration corresponds
  to a different stage in the MoDL network and updates the prediction by solving

  $$\mathbf{x}^{k+1} = (A^T A + \lambda \, \mathbf{I})^{-1} (A^T \mathbf{y} + \lambda \, \mathbf{z}^k) \;,$$

via conjugate gradient. In the expression, $k$ is the index of the stage (iteration),
 $\mathbf{z}^k = \mathrm{ResNet}(\mathbf{x}^{k})$ is the regularization
 (a denoiser implemented as a residual convolutional neural network), $\mathbf{x}^k$ is the output
  of the previous stage, $\lambda > 0$
  is a learned regularization parameter, and $\mathbf{I}$ is the identity operator.
  The output of the final stage is the reconstructed image.
"""

"""
## Constructing the Data Sets

Machine learning processing for images in Flax assumes the following data shape: $(K, H, W, C)$

- $K$ is the number of image samples
- $H$ is the height of the images
- $W$ is the width of the images
- $C$ is the number of channels of the images (e.g. 1 for grayscale images, 3 for color images)

**Reformat the training data to have the expected Flax shape.**
"""

# startq
foam_collection = foam_collection.reshape(...)
sino_collection = sino_collection.reshape(...)
# starta
foam_collection = foam_collection.reshape((nfoams, N, N, 1))
sino_collection = sino_collection.reshape((nfoams, n_projection, N, 1))
# endqa

"""
SCICO passes the data to the ML models as a dictionary, with the `image` key to define the input and the `label` key to define the expected output. In other words, (`image`, `label`) define the pair needed for supervised training.

**Construct training and testing partitions for the CT reconstuction problem**. Use the first 16 images for training and the rest for testing. Remember that in the CT problem you want to reconstruct images from sinograms.
"""

# startq
train_ds = {}
test_ds = {}
# starta
train_ds = {"image": sino_collection[:16], "label": foam_collection[:16]}
test_ds = {"image": sino_collection[16:], "label": foam_collection[16:]}
# endqa

"""
## Configuring the ML model and its training

SCICO configures both model and training via dictionaries too. An example of configuration dictionaries with the corresponding definitions is shown next.

Run the next cell to build the configuration dictionaries.
"""
# Model configuration
model_conf = {
    "depth": 2,  # Number of layers (=iterations) in the unrolled ML model
    "num_filters": 16,  # Number of filters in the denoiser
    "block_depth": 3,  # Number of layers in the denoiser
}

# Training configuration
train_conf = {
    "seed": 100,  # Seed for random generation
    "opt_type": "ADAM",  # Optimization (other available options: SGD, ADAMW)
    "batch_size": 8,  # Number of samples to include in each batch
    "num_epochs": 50,  # Number of training epochs
    "base_learning_rate": 1e-2,  # Base learning rate
    "warmup_epochs": 0,  # Iterations to reach the base learning rate (if a scheduler is specified)
    "log_every_steps": 5,  # Frequency of reporting training stats, given in units of training steps
    "checkpointing": False,  # Checkpoint stats during training
    "log": True,  # Display training messages and statistics
}

"""
## Constructing the ML model

SCICO ML functionality is based on FLAX (see https://flax.readthedocs.io/en/latest/overview.html). Frequently used models are provided in SCICO.

Run the next cell to import the Flax functionality in SCICO.
"""

from scico import flax as sflax

"""
**Look how to construct a MoDL model in the SCICO documentation and construct it.** Use the parameters already defined, but use 1 as depth.

"""
# startq
channels = ...
model = sflax.MoDLNet(...)
# starta
channels = train_ds["image"].shape[-1]
model = sflax.MoDLNet(
    operator=A,
    depth=1,
    channels=channels,
    num_filters=model_conf["num_filters"],
    block_depth=model_conf["block_depth"],
    cg_iter=3,
)
# endqa

"""
You are done with part 2. Please report back in the Webex chat: **done with part 2**.

While you wait for others to finish, explore other ML models available in SCICO.

🛑 **PAUSE HERE** 🛑
"""

"""
# Training the MoDL model

$\lambda$, the regularization parameter in MoDL, is also learned in the training process. However, it is important that it remains positive.

Run the next cell to build the structure necessary to assure that the training will respect such constraint.
"""

from functools import partial

from scico.flax.train.train import clip_positive, construct_traversal

lmbdatrav = construct_traversal(
    "lmbda"
)  # Functionality to get parameter to constraint inside model
lmbdapos = partial(
    clip_positive,  # Type of constraint to apply, here positivity constraint
    traversal=lmbdatrav, # Operate over lmbda parameters
    minval=5e-4,  # Minimum value to accept when enforcing the positivity constraint
)

train_conf["post_lst"] = [lmbdapos]  # Constraints to model parameters


"""
Now that we have all the structures needed for training: a data set, a model, and parameter constraints, we can use SCICO to train the model.

All the training in SCICO is carried out through the `BasicFlaxTrainer` class. The following cell shows how to instantiate an object of that class. It uses an MSE loss function as minimization criterion by default (so it is not necessary to pass it explicitly). Look into the documentation and compare with the arguments provided.

Run the next cell to train the model for the number of epochs specified. Check the output being produced. It corresponds, first, to the variables of the model and, next, to the training statistics. If it is taking too long, you can try to train for less epochs or use less layers in the model.
"""
from time import time

trainer = sflax.BasicFlaxTrainer(
    train_conf,  # Dictionary with training configuration
    model,  # Model to train
    train_ds,  # Data set for training (image-label dictionary)
    test_ds,  # Data set for testing (image-label dictionary)
)
start_time = time()
modvar, stats_object_ini = trainer.train()
time_train = time() - start_time
print(f"Time train [s]: {time_train}")

"""
Run the next cell to plot the training statistics.
"""
hist = stats_object_ini.history(transpose=True)
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 5))
plot.plot(
    np.vstack((hist.Train_Loss, hist.Eval_Loss)).T,
    x=hist.Epoch,
    ptyp="semilogy",
    title="Loss function",
    xlbl="Epoch",
    ylbl="Loss value",
    lgnd=("Train", "Test"),
    fig=fig,
    ax=ax[0],
)

plot.plot(
    np.vstack((hist.Train_SNR, hist.Eval_SNR)).T,
    x=hist.Epoch,
    title="Metric",
    xlbl="Epoch",
    ylbl="SNR (dB)",
    lgnd=("Train", "Test"),
    fig=fig,
    ax=ax[1],
)
fig.show()

"""
The MoDL architecture shares the parameters between the different iteration layers. The previous training was the initialization and used only one layer (corresponding to only unrolling one iteration of the optimization computation). Now we can train the model with the specified depth (from `model_conf`).

**Repeat the training process**, but this time use the configured depth, 10 cg iterations and initialize with the current model parameters. Train for 10 epochs.
In addition, set an exponentially decaying learning rate by
 adding a `create_lr_schedule` and a decay rate of 0.95 to the training configuration dictionary. Create a new trainer object. Make sure you pass the parameter `variables0=modvar` when initializing the object to start training with your pretrained weights.
"""

# startq
from scico.flax.train.train import create_exp_lr_schedule

model.depth = ...
model.cg_iter = ...
train_conf["num_epochs"] = ...
train_conf["lr_decay_rate"] = ...
train_conf["create_lr_schedule"] = ...
train_conf["post_lst"] = ...

trainer = ...
start_time = time()
modvar, stats_object = ...
time_train = time() - start_time
print(f"Time train [s]: {time_train}")

# starta
from scico.flax.train.train import create_exp_lr_schedule

model.depth = model_conf["depth"]
model.cg_iter = 10
train_conf["num_epochs"] = 10
train_conf["lr_decay_rate"] = 0.95
train_conf["create_lr_schedule"] = create_exp_lr_schedule  # Exponentially decaying LR
train_conf["post_lst"] = [lmbdapos]  # Constraints to model parameters"]

trainer = sflax.BasicFlaxTrainer(
    train_conf,  # Dictionary with training configuration
    model,  # Model to train
    train_ds,  # Data set for training (image-label dictionary)
    test_ds,  # Data set for testing (image-label dictionary)
    variables0=modvar,  # Model variables after initial training
)
start_time = time()
modvar, stats_object = trainer.train()
time_train = time() - start_time
print(f"Time train [s]: {time_train}")
# endqa

"""
Plot the training stats for this last training.
"""

# startq
...
# starta
hist = stats_object.history(transpose=True)
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 5))
plot.plot(
    np.vstack((hist.Train_Loss, hist.Eval_Loss)).T,
    x=hist.Epoch,
    ptyp="semilogy",
    title="Loss function",
    xlbl="Epoch",
    ylbl="Loss value",
    lgnd=("Train", "Test"),
    fig=fig,
    ax=ax[0],
)

plot.plot(
    np.vstack((hist.Train_SNR, hist.Eval_SNR)).T,
    x=hist.Epoch,
    title="Metric",
    xlbl="Epoch",
    ylbl="SNR (dB)",
    lgnd=("Train", "Test"),
    fig=fig,
    ax=ax[1],
)

fig.show()
# endqa

"""
You are done with part 3. Please report back in the Webex chat: **done with part 3**.

While you wait for others to finish, think of things you could try
 to improve the performance of MoDL for CT reconstruction.

🛑 **PAUSE HERE** 🛑
"""

"""
# Deploying a trained model

If all you need to do is to apply a trained model, SCICO provides the `FlaxMap` class. This also allows you to connect trained models to other SCICO functionality.

Run the next cell to see the trained model applied to the testing set.
"""
start_time = time()
fmap = sflax.FlaxMap(model, modvar)
output = fmap(test_ds["image"])
time_eval = time() - start_time
output = np.clip(output, a_min=0, a_max=1.0)

"""
Use the SCICO documentation to figure out how to **compute SNR and MAE for the reconstructions obtained with MoDL**.
You might start looking in https://scico.readthedocs.io/en/latest/_autosummary/scico.metric.html.
"""
# startq
from scico import metric

snr_eval = ...
mae_eval = ...
print(f"SNR [dB]: {snr_eval}")
print(f"MAE: {mae_eval}")
# starta
from scico import metric

snr_eval = metric.snr(test_ds["label"], output)
mae_eval = metric.mae(test_ds["label"], output)
print(f"SNR [dB]: {snr_eval}")
print(f"MAE: {mae_eval}")
# endqa

"""
Run the next cell to check one of the testing results.
"""
np.random.seed(543)
indx = np.random.randint(0, high=8)

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(test_ds["label"][indx, ..., 0], title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    test_ds["image"][indx, ..., 0],
    title="Sinogram",
    cbar=None,
    fig=fig,
    ax=ax[1],
)
plot.imview(
    output[indx, ..., 0],
    title="MoDL Reconstruction\nSNR: %.2f (dB), MAE: %.3f"
    % (
        metric.snr(test_ds["label"][indx, ..., 0], output[indx, ..., 0]),
        metric.mae(test_ds["label"][indx, ..., 0], output[indx, ..., 0]),
    ),
    fig=fig,
    ax=ax[2],
)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[2].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

"""
You are done with part 4. Please report back in the Webex chat: **done with part 4**.

While you wait for others to finish, think why you need to provide both model and variables when constructing the FlaxMap class.

🛑 **PAUSE HERE** 🛑
"""

"""
A basic UQ/ML model is to perform aletoric analysis and assume that the output (predictions) are independently and normally distributed. We can build a model to simultaneosuly predict mean and variance of the (independent) outputs.

The corresponding loss function, denominated a heteroscedastic loss, can be expressed as

$$L_{\mathrm{het}} = \frac{1}{N} \sum_i \frac{1}{2 \sigma(\mathbf{x_i})^2} || \mathbf{y_i} - f(\mathbf{x_i}) ||^2 \frac{1}{2} \log \sigma(\mathbf{x_i})^2;,$$

with $f(\mathbf{x_i})$ the mean prediction and $\sigma(\mathbf{x_i})^2$ the variance.

Run the next cell to define the heteroscedastic loss function.
"""
import jax.numpy as jnp

def het_loss(predictions, targets):
    diff_sq = (targets - predictions[:,...[0]])**2
    log_sig2 = predictions[:,...[1]])

    return jnp.mean(jnp.exp(-log_sig2) * diff_sq + log_sig2)

"""
Build a MoDL model that can use the heteroscedastic loss function and train it as you did with previous models.
"""


# startq
from flax.linen.module import Module, compact
from scico.flax import ResNet
from scico.flax.inverse import cg_solver
from typing import Any, Tuple

class MoDLNet_het(Module): ...

model = MoDLNet_het(...)
train_conf = {...}

trainer = sflax.BasicFlaxTrainer(...)
start_time = time()
modvar, stats_object = trainer.train()
time_train = time() - start_time
print(f"Time train [s]: {time_train}")

# starta
from flax.linen.module import Module, compact
from scico.flax import ResNet
from scico.flax.inverse import cg_solver
from typing import Any, Tuple

class MoDLNet_het(Module):
    operator: Any
    depth: int
    channels: int
    num_filters: int
    block_depth: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    lmbda_ini: float = 0.5
    dtype: Any = jnp.float32
    cg_iter: int = 10

    @compact
    def __call__(self, y: Array, train: bool = True) -> Array:
        """Apply MoDL net for inversion.

        Args:
            y: The nd-array with signal to invert.
            train: Flag to differentiate between training and testing stages.

        Returns:
            The mean of the reconstructed signal and the predicted variance.
        """

        def lmbda_init_wrap(rng: PRNGKey, shape: Shape, dtype: DType = self.dtype) -> Array:
            return jnp.ones(shape, dtype) * self.lmbda_ini

        lmbda = self.param("lmbda", lmbda_init_wrap, (1,))

        resnet = ResNet(
            self.block_depth,
            self.channels * 2, # het
            self.num_filters,
            self.kernel_size,
            self.strides,
            dtype=self.dtype,
        )

        ah_f = lambda v: jnp.atleast_3d(self.operator.adj(v.reshape(self.operator.output_shape)))

        Ahb = lax.map(ah_f, y)
        x = jnp.repeat(Ahb, 2, axis=-1)

        ahaI_f = lambda v: self.operator.adj(self.operator(v)) + lmbda * v

        cgsol = lambda b: jnp.atleast_3d(
            cg_solver(ahaI_f, b.reshape(self.operator.input_shape), maxiter=self.cg_iter)
        )

        for i in range(self.depth):
            z = resnet(x, train)
            # Solve:
            # (AH A + lmbda I) x = Ahb + lmbda * z
            b = Ahb + lmbda * z[...,[0]]
            x0 = lax.map(cgsol, b)
            b = Ahb + lmbda * z[...,[1]]
            x1 = lax.map(cgsol, b)
            x = jnp.concatenate((x0, x1), axis=-1)
        return x

model = MoDLNet_het(
    operator=A,
    depth=1,
    channels=channels,
    num_filters=model_conf["num_filters"],
    block_depth=model_conf["block_depth"],
)

train_conf = {
    "seed": 100,  # Seed for random generation
    "opt_type": "ADAM",  # Optimization (other available options: SGD, ADAMW)
    "batch_size": 8,  # Number of samples to include in each batch
    "num_epochs": 50,  # Number of training epochs
    "base_learning_rate": 1e-2,  # Base learning rate
    "warmup_epochs": 0,  # Iterations to reach the base learning rate (if a scheduler is specified)
    "log_every_steps": 5,  # Frequency of reporting training stats, given in units of training steps
    "checkpointing": False,  # Checkpoint stats during training
    "log": True,  # Display training messages and statistics
    "post_lst": [lmbdapos]  # Constraints to model parameters
    "criterion": het_loss # Minimize heteroscedastic loss
}

trainer = sflax.BasicFlaxTrainer(
    train_conf,  # Dictionary with training configuration
    model,  # Model to train
    train_ds,  # Data set for training (image-label dictionary)
    test_ds,  # Data set for testing (image-label dictionary)
)
start_time = time()
modvar, stats_object = trainer.train()
time_train = time() - start_time
print(f"Time train [s]: {time_train}")
# endqa


"""
Run the next cell to check one of the testing results.
"""
fmap = sflax.FlaxMap(model, modvar)
output = fmap(test_ds["image"])

np.random.seed(543)
indx = np.random.randint(0, high=8)
# extract standard deviation
std_indx = jnp.sqrt(jnp.exp(output[indx, ..., 1]))

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(test_ds["label"][indx, ..., 0], title="Ground truth", cbar=None, fig=fig, ax=ax[0])

plot.imview(
    output[indx, ..., 0],
    title="MoDL Reconstruction\nSNR: %.2f (dB), MAE: %.3f"
    % (
        metric.snr(test_ds["label"][indx, ..., 0], output[indx, ..., 0]),
        metric.mae(test_ds["label"][indx, ..., 0], output[indx, ..., 0]),
    ),
    fig=fig,
    ax=ax[1],
)
plot.imview(std_indx, title="Standard Deviation", cbar=None, fig=fig, ax=ax[2])

divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[2].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

"""
## Conclusion
This tutorial has shown how to set up
a SCICO pipeline for performing computed tomography (CT) reconstruction using data-driven priors. In doing so, it has demonstrated SCICO functionality to build and train ML models for imaging problems. The functionality is based on FLAX and provides a  straightforward pipeline for other ML applications.
"""

"""
You are done with this tutorial! Please report back in the Webex chat: **done with the CNN tutorial**.

While you wait for others to finish, you could think of similar problems you may want to solve with SCICO.
We would be happy to talk with you about using SCICO in your own work!
"""
