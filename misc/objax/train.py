#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functionality to train DnCNN model in objax.
"""

import os
import time

import numpy as np

import jax
import jax.numpy as jnp

import objax
import objax_model as objxm
import utils as dtu
from absl import app, flags, logging

flags.DEFINE_integer("depth", 6, "Number of layers in model")
flags.DEFINE_integer("num_filters", 64, "Number of filters in model")

flags.DEFINE_string("path", "~/data/BSDS/train_CBSD500/set400/", "Train data directory.")
flags.DEFINE_string("ext", "jpg", "Data extension.")
flags.DEFINE_integer("output_size", 40, "Size of images")
flags.DEFINE_integer("stride", 16, "Stride for sampling patches")
flags.DEFINE_boolean("run_gray", False, "Run grayscale experiment")
flags.DEFINE_integer("num_img", 200, "Number of images for training")
flags.DEFINE_string("test_path", "~/data/BSDS/test_CBSD68/original/", "Test data directory.")
flags.DEFINE_string("test_ext", "jpg", "Testing data extension.")
flags.DEFINE_integer("test_numimg", 10, "Number of images for testing")
flags.DEFINE_string("data_mode", "dn", "Type of data corruption.")
flags.DEFINE_float("noise_level", 0.1, "STD for Gaussian noise.")
flags.DEFINE_boolean("noise_range", False, "Train with range of noise")
flags.DEFINE_float("test_split", 0.1, "Fraction of data for testing.")

flags.DEFINE_float("base_learning_rate", 0.1, "Base learning rate.")
flags.DEFINE_integer("train_device_batch_size", 16, "Per-device training batch size")
flags.DEFINE_integer("eval_device_batch_size", 32, "Per-device eval batch size.")
flags.DEFINE_string("model_dir", "./output/", "Model directory.")
flags.DEFINE_integer("num_train_epochs", 10, "Number of epochs for training")
flags.DEFINE_integer("eval_every_n_steps", 400, "How often to run eval.")
flags.DEFINE_integer(
    "max_eval_batches",
    -1,
    "Maximum number of batches used for "
    "evaluation, zero or negative number means use all batches.",
)

flags.DEFINE_string("opt_type", "SGD", "Type of optimizer SGD with momentum or ADAM")
flags.DEFINE_string(
    "lrsc_type", "EXP", "Type of learning rate decay: " "FIX: fixed, EXP: exponential"
)

FLAGS = flags.FLAGS


@jax.jit
def psnr_jnp(vref, vcmp):
    """PSNR computation assuming signal
    range [0,1]
    """
    mse_ = jnp.mean((vref - vcmp) ** 2)
    rt = 1.0 / mse_
    return 10.0 * jnp.log10(rt)


class Experiment:
    """Class with all code to run experiment."""

    def __init__(self, channels):
        # Some constants
        self.total_batch_size = FLAGS.train_device_batch_size * jax.device_count()
        self.base_learning_rate = FLAGS.base_learning_rate  # * self.total_batch_size / 32
        # Create model
        self.model = objxm.DnCNN_Net(FLAGS.depth, channels, FLAGS.num_filters)
        self.model_vars = self.model.vars()
        print(self.model_vars)
        # Create parallel eval op
        self.evaluate_batch_parallel = objax.Parallel(
            self.evaluate_batch, self.model_vars, reduce=lambda x: x.sum(0)
        )
        # Base for learning rate decay
        if FLAGS.lrsc_type == "EXP":
            self.gamma = np.exp(np.log(1.0e3) / -FLAGS.num_train_epochs)
            print(f"Base for learning rate decay: {self.gamma}")
            print(f"Initial learning rate: {self.base_learning_rate}")
        else:
            self.gamma = 1.0
        # Create parallel training op
        if FLAGS.opt_type == "SGD":
            self.optimizer = objax.optimizer.Momentum(self.model_vars, momentum=0.9, nesterov=True)
            # self.optimizer = objax.optimizer.Momentum(self.model_vars, momentum=0.9)
        else:
            self.optimizer = objax.optimizer.Adam(self.model_vars)
        self.compute_grads_loss = objax.GradValues(self.loss_fn, self.model_vars)
        self.all_vars = self.model_vars + self.optimizer.vars()
        self.train_op_parallel = objax.Parallel(self.train_op, self.all_vars, reduce=lambda x: x[0])
        # Summary writer
        self.summary_writer = objax.jaxboard.SummaryWriter(os.path.join(FLAGS.model_dir, "tb"))

    def evaluate_batch(self, images, labels):
        output = self.model(images, training=False)
        psnr = jnp.sum(jax.vmap(psnr_jnp)(labels, output), axis=0)
        return psnr

    def run_eval(self, test_ds_iter):
        """Runs evaluation and returns mean PSNR."""
        psnr_pred = 0.0
        total_examples = 0
        for batch_index, batch in enumerate(test_ds_iter):

            psnr_pred += self.evaluate_batch_parallel(batch["images"], batch["labels"])
            total_examples += batch["images"].shape[0]
            if (FLAGS.max_eval_batches > 0) and (batch_index + 1 >= FLAGS.max_eval_batches):
                break

        return psnr_pred / total_examples

    def loss_fn(self, images, labels):
        """Computes loss function.

        Args:
          images: tensor with images NCHW
          labels: tensors with dense labels, shape (batch_size,)

        Returns:
          Tuple (total_loss, losses_dictionary).
        """
        output = self.model(images, training=False)
        mse_loss = objax.functional.loss.mean_squared_error(labels, output).mean()
        return mse_loss, {"total_loss": mse_loss}

    def learning_rate(self, epoch: float):
        """Computes exponentially decaying learning rate."""

        lr = self.base_learning_rate * self.gamma ** (epoch)

        return lr

    def train_op(self, images, labels, cur_epoch):
        cur_epoch = cur_epoch[0]  # because cur_epoch is array of size 1
        grads, (_, losses_dict) = self.compute_grads_loss(images, labels)
        grads = objax.functional.parallel.pmean(grads)
        losses_dict = objax.functional.parallel.pmean(losses_dict)
        learning_rate = self.learning_rate(cur_epoch)
        self.optimizer(learning_rate, grads)
        return dict(**losses_dict, learning_rate=learning_rate, epoch=cur_epoch)

    def train_and_eval(self, train_ds, test_ds):
        """Runs training and evaluation."""
        steps_per_epoch = train_ds["images"].shape[0] / self.total_batch_size
        total_train_steps = int(steps_per_epoch * FLAGS.num_train_epochs)
        eval_every_n_steps = FLAGS.eval_every_n_steps

        checkpoint = objax.io.Checkpoint(FLAGS.model_dir, keep_ckpts=10)
        start_step, _ = checkpoint.restore(self.all_vars)
        cur_epoch = np.zeros([jax.local_device_count()], dtype=np.float32)

        batch_tr_size = jax.local_device_count() * FLAGS.train_device_batch_size
        batch_eval_size = jax.local_device_count() * FLAGS.eval_device_batch_size
        rng = jax.random.PRNGKey(0)
        train_ds_iter = dtu.IterateData(train_ds, batch_tr_size, True, rng)
        test_ds_iter = dtu.IterateData(test_ds, batch_eval_size)

        for big_step in range(start_step, total_train_steps, eval_every_n_steps):
            print(f"Running training steps {big_step + 1} - {big_step + eval_every_n_steps}")
            with self.all_vars.replicate():
                # training
                start_time = time.time()
                for cur_step in range(big_step + 1, big_step + eval_every_n_steps + 1):
                    batch = next(train_ds_iter)
                    cur_epoch[:] = cur_step / steps_per_epoch
                    monitors = self.train_op_parallel(batch["images"], batch["labels"], cur_epoch)
                elapsed_train_time = time.time() - start_time
                # eval
                start_time = time.time()
                psnr = self.run_eval(test_ds_iter)
                elapsed_eval_time = time.time() - start_time
            # In multi-host setup only first host saves summaries and checkpoints.
            if jax.host_id() == 0:
                # save summary
                summary = objax.jaxboard.Summary()
                for k, v in monitors.items():
                    summary.scalar(f"train/{k}", v)
                # # Uncomment following two lines to save summary with training images
                # summary.image('input/train_img',
                #               imagenet_data.normalize_image_for_view(batch['images'][0]))
                summary.scalar("test/psnr", psnr)
                self.summary_writer.write(summary, step=cur_step)
                # save checkpoint
                checkpoint.save(self.all_vars, cur_step)
            # print info
            print(
                "Step %d -- Epoch %.2f -- LR: %.2e -- Loss %.4f  PSNR %.2f"
                % (
                    cur_step,
                    cur_step / steps_per_epoch,
                    monitors["learning_rate"],
                    monitors["total_loss"],
                    psnr,
                )
            )
            print(
                "    Training took %.1f seconds, eval took %.1f seconds"
                % (elapsed_train_time, elapsed_eval_time),
                flush=True,
            )


def main(argv):
    del argv
    print("JAX host: %d / %d" % (jax.host_id(), jax.host_count()))
    print("JAX devices:\n%s" % "\n".join(str(d) for d in jax.devices()), flush=True)

    # objax requires channel before height and width
    train_ds, test_ds = dtu.build_img_dataset(FLAGS, channel_first=True)
    numtrain = train_ds["images"].shape[0]
    numtest = test_ds["images"].shape[0]
    channels = train_ds["images"].shape[1]
    print(f"Channels: {channels}")
    print(f"Num training images: {FLAGS.num_img}")
    print(f"Num training patches: {numtrain}")
    print(f"Num test patches: {numtest}")
    print(f"patch_size: {train_ds['images'].shape[2:]}")
    # Input noise
    inpsnr = jnp.mean(jax.vmap(psnr_jnp)(train_ds["labels"], train_ds["images"]), axis=0)
    print(f"input psnr: {inpsnr}")

    experiment = Experiment(channels=channels)
    experiment.train_and_eval(train_ds, test_ds)
    objax.util.multi_host_barrier()


if __name__ == "__main__":
    logging.set_verbosity(logging.ERROR)
    jax.config.config_with_absl()
    app.run(main)
