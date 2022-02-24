#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flax learning rate functionality.
"""

import optax

def create_learning_rate_fn(
    config: ml_collections.ConfigDict, base_learning_rate: float, steps_per_epoch: int
):
    """Create learning rate schedule."""
    #warmup_fn = optax.linear_schedule(
    #    init_value=0.0,
    #    end_value=base_learning_rate,
    #    transition_steps=config.warmup_epochs * steps_per_epoch,
    #)
    ### expon_epochs = max(config.warmup_epochs * steps_per_epoch, 1)
    ##expon_epochs = config.num_epochs * steps_per_epoch
    ##decay_rate = jnp.exp(jnp.log(1e2) / -expon_epochs)
    ##expon_fn = optax.exponential_decay(
    ##    init_value=base_learning_rate,
    ##    transition_steps=expon_epochs,
    ##    decay_rate=decay_rate,
    ##)
    #cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    #cosine_fn = optax.cosine_decay_schedule(
    #    init_value=base_learning_rate,
    #    decay_steps=cosine_epochs * steps_per_epoch)
    #schedule_fn = optax.join_schedules(
    #    schedules=[warmup_fn, cosine_fn],
    #    boundaries=[config.warmup_epochs * steps_per_epoch],
    #)
    schedule_fn = optax.constant_schedule(base_learning_rate)
    return schedule_fn

def compute_spectral_normalization(params, xshp):
    # Unfreeze params to normal dict.
    prms = unfreeze(params)
    # Get flattened-key: value list.
    flat_prms = {'/'.join(k): v for k, v in flax.traverse_util.flatten_dict(prms).items()}
    # Normalize kernel by spectral norm
    for k in flat_prms.keys():
        k_tstr = k.split('/')
        prm_str = k_tstr[-1]
        if prm_str == 'kernel':
            sgm_v = norm.spectral_norm_conv2d(flat_prms[k], xshp)
            flat_prms[k] /= (sgm_v * 1.02)
    # Unflatten.
    unflat_prms = flax.traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in flat_prms.items()})
    # Refreeze.
    unflat_prms = freeze(unflat_prms)

    return unflat_prms
