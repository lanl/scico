"""
# Example: Data-Driven Priors for Inverse Problems

In this example,
 we will explore how to use CNN models as data-driven priors in a SCICO pipeline for performing computed tomography (CT) reconstruction.

## Introduction

Suppose that you are performing CT scans of many of similar objects
 and want to construct a pipeline to rapidly compute the reconstruction
 of each new measurement.
For this example, we will use computer-generated foam images
as the objects we want to image.
Run the next cell to generate and visualize one of such foam.
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

**Define an ASTRA SCICO CT projector assuming 45 equally spaced projections.**
"""
# startq

n_projection = ...  # number of projections
angles = ...
A = ...
# starta
from scico.linop.radon_astra import TomographicProjector

n_projection = 45  # number of projections
angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
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
Let's use parallel computation to accelerate the sinogram generation.
Distributing processing among GPUs on the same node happens automatically in JAX,
 but for CPUs, JAX only uses one core by default.
The following commands force JAX to use 8 CPU cores.
 (If GPUs are available, the command will be ignored)
"""

import os

import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
platform = jax.lib.xla_bridge.get_backend().platform
print("Platform: ", platform)

"""
For purely jax functionality, a distributed processing can be computed via `jax.vmap`. However, the CT operator uses a python (not JAX) library.
In that case we can distribute the processing via `jax.lax`. Run the next cell to distribute the computation of the sinograms.
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

Run the next cell to build the configuration dictionary.
"""
model_conf = {
    "depth": 2,  # Number of layers (=iterations) in the unrolled ML model
    "num_filters": 16,  # Number of filters in the denoiser
    "block_depth": 3,  # Number of layers in the denoiser
}

train_conf = {
    "seed": 100,  # Seed for random generation
    "opt_type": "ADAM",  # Optimization (other available options: SGD, ADAMW)
    "batch_size": 8,  # Number of samples to include in each batch
    "num_epochs": 50,  # Number of training epochs
    "base_learning_rate": 1e-2,  # Base learning rate
    "warmup_epochs": 0,  # Iterations to reach the base learning rate (if a scheduler is specified)
    "log_every_steps": 5,  # Frequency of reporting training stats, given in units of training steps
    "workdir": "./modl_ct/",
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

All the training in SCICO is carried out through the `train_and_evaluate` function. The following cell shows how to pass the necessary information to that function. It uses an MSE loss function as minimization criterion by default. Look into the documentation and compare with the arguments provided.

Run the next cell to train the model for the number of epochs specified. Check the output being produced. It corresponds, first, to the description of the model architecture and parameter default initialization and, next, to the training statistics. If it is taking too long, you can try to train for less epochs or use less layers in the model.
"""
from time import time

trainer = sflax.train_and_evaluate(
    train_conf,  # Dictionary with training configuration
    model,  # Model to train
    train_ds,  # Data set for training (image-label dictionary)
    test_ds,  # Data set for testing (image-label dictionary)
)
start_time = time()
modvar, stats_object = trainer.train()
time_train = time() - start_time
print(f"Time train [s]: {time_train}")

"""
Run the next cell to plot the training statistics.
"""
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

"""
The MoDL architecture shares the parameters between the different iteration layers. The previous training was the initialization and used only one iteration. Now we can train the model with the specified depth (from `dconf`).

**Repeat the training process**, but this time use the configured depth, 10 cg iterations and initialize with the current model parameters. Train for 100 epochs.
In addition, set an exponentially decaying learning rate by
 adding a decay rate of 0.95 to the configuration dictionary.
and using the `create_lr_schedule` option for `train_and_evaluate`.
Make sure you pass the parameter `variables0=modvar` to `train_and_evaluate`
to start with your pretrained weights.
"""

# startq
from scico.flax.train.train import create_exp_lr_schedule

model.depth = ...
model.cg_iter = ...
dconf["num_epochs"] = ...
dconf["lr_decay_rate"] = ...

workdir2 = workdir + "iterated/"

start_time = time()
modvar, stats_object = sflax.train_and_evaluate(...)
time_train = time() - start_time
print(f"Time train [s]: {time_train}")

# starta
from scico.flax.train.train import create_exp_lr_schedule

model.depth = dconf["depth"]
model.cg_iter = 10
dconf["num_epochs"] = 100
dconf["lr_decay_rate"] = 0.95

workdir2 = workdir + "iterated/"

start_time = time()
modvar, stats_object = sflax.train_and_evaluate(
    dconf,  # Dictionary with training configuration
    workdir2,  # Directory to store checkpoints
    model,  # Model to train
    train_ds,  # Data set for training (image-label dictionary)
    test_ds,  # Data set for testing (image-label dictionary)
    create_lr_schedule=create_exp_lr_schedule,  # Exponentially decaying LR
    post_lst=[lmbdapos],  # Constraints to model parameters
    variables0=modvar,  # Model variables after initial training
    checkpointing=False,  # Checkpoint stats during training
    log=True,  # Display training messages and statistics
)
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
Use the SCICO documentation to figure out how to compute SNR and MAE for the reconstructions obtained with MoDL.
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
## Conclusion
This tutorial has shown how to set up
a SCICO pipeline for performing computed tomography (CT) reconstruction using data-driven priors. In doing so, it has demonstrated SCICO functionality to build and train ML models for imaging problems. The functionality is based on FLAX and provides a  straightforward pipeline for other ML applications.
"""

"""
You are done with this tutorial! Please report back in the Webex chat: **done with the CNN tutorial**.

While you wait for others to finish, you could think of similar problems you may want to solve with SCICO.
We would be happy to talk with you about using SCICO in your own work!
"""
