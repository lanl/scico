#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Useful metrics.
"""

import jax.numpy as jnp
from jax import lax
import optax


def psnr_jnp(vref, vcmp):
    mse_ = jnp.mean((vref-vcmp)**2)
    rt = 1. / mse_
    return 10. * jnp.log10(rt)


def mse_loss(output, labels):
    mse = optax.l2_loss(output, labels)
    return jnp.mean(mse)


def compute_metrics(output, labels):
    loss = mse_loss(output, labels)
    psnr = psnr_jnp(labels, output)
    metrics = {
        'loss': loss,
        'psnr': psnr,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics
