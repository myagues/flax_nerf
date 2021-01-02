import functools

import jax

from absl import logging
from jax import numpy as jnp, lax, random

from rays_utils import prepare_rays, raw2outputs, render_rays, render_rays_fine


psnr_fn = lambda x: -10.0 * jnp.log(x) / jnp.log(10.0)


def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=8000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000,
    staircase=False,
):
    """Creates learning rate schedule.
    Interprets factors in the factors string which can consist of:
    * constant: interpreted as the constant value,
    * linear_warmup: interpreted as linear warmup until warmup_steps,
    * rsqrt_decay: divide by square root of max(step, warmup_steps)
    * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
    * decay_every: Every k steps decay the learning rate by decay_factor.
    * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
    Args:
        factors: string, factors separated by '*' that defines the schedule.
        base_learning_rate: float, the starting constant for the lr schedule.
        warmup_steps: int, how many steps to warm up for in the warmup schedule.
        decay_factor: float, the amount to decay the learning rate by.
        steps_per_decay: int, how often to decay the learning rate.
        steps_per_cycle: int, steps per cycle when using cosine decay.
    Returns:
        a function learning_rate(step): the step-dependent lr.
    """
    factors = [n.strip() for n in factors.split("*")]

    def step_fn(step):
        """Step to learning rate function."""
        ret = 1.0
        for name in factors:
            if name == "constant":
                ret *= base_learning_rate
            elif name == "linear_warmup":
                ret *= jnp.minimum(1.0, step / warmup_steps)
            elif name == "rsqrt_decay":
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == "rsqrt_normalized_decay":
                ret *= jnp.sqrt(warmup_steps)
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == "decay_every":
                if staircase:
                    decay = step // steps_per_decay
                else:
                    decay = step / steps_per_decay
                ret *= decay_factor ** decay
            elif name == "cosine_decay":
                progress = jnp.maximum(
                    0.0, (step - warmup_steps) / float(steps_per_cycle)
                )
                ret *= jnp.maximum(
                    0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0)))
                )
            else:
                raise ValueError(f"Unknown factor {name}.")
        return jnp.asarray(ret, dtype=jnp.float32)

    return step_fn


def train_step(
    model_fn,
    config,
    lr_fn,
    hwfnf,
    state,
    batch,
    coords=None,
    rng=None,
):
    """Perform a single training step."""
    rng_0, rng_1, rng_2, rng_3, rng_4 = random.split(rng, 5)
    inputs, target = batch
    hwf, near, far = hwfnf
    apply_coarse, apply_fine = model_fn
    opt_coarse, opt_fine = state.optimizer_coarse, state.optimizer_fine

    if not config.batching:
        rays = prepare_rays(None, hwf, config, near, far, inputs[:3, :4], None)
        if coords is None:
            coords = jnp.meshgrid(jnp.arange(hwf[0]), jnp.arange(hwf[1]), indexing="ij")
            coords = jnp.stack(coords, axis=-1).reshape([-1, 2])
        select_idx = random.choice(
            rng_0,
            coords.shape[0],
            shape=[config.num_rand],
            replace=False,
        )
        select_idx = coords[select_idx]
        rays = rays[select_idx[:, 0], select_idx[:, 1]]
        target = target[select_idx[:, 0], select_idx[:, 1]]
    else:
        rays = inputs

    *rays, viewdirs = jnp.split(rays, [3, 6, 7, 8], axis=-1)
    raw2outputs_ = functools.partial(
        raw2outputs,
        raw_noise_std=config.raw_noise_std,
        white_bkgd=config.white_bkgd,
    )

    def loss_fn(params_coarse, params_fine=None):
        """Loss function used for training."""
        pts, z_vals = render_rays(rays, config, rng_1)
        raw_c = apply_coarse({"params": params_coarse}, pts, viewdirs).reshape(
            [config.num_rand, config.num_samples, 4]
        )
        coarse_res, weights = raw2outputs_(raw_c, z_vals, rays[1], rng=rng_2)
        loss_c = jnp.mean((coarse_res["rgb"] - target) ** 2.0)

        loss_f = 0
        if config.num_importance > 0:
            pts, z_vals, _ = render_rays_fine(
                rays[:2], z_vals, weights, config.num_importance, config.perturb, rng_3
            )
            raw_f = apply_fine({"params": params_fine}, pts, viewdirs).reshape(
                [config.num_rand, config.num_samples + config.num_importance, 4]
            )
            fine_res, _ = raw2outputs_(raw_f, z_vals, rays[1], rng=rng_4)
            loss_f = jnp.mean((fine_res["rgb"] - target) ** 2.0)

        loss = loss_c + loss_f
        return loss, (loss_c, loss_f)

    lr = lr_fn(state.step)
    if config.num_importance > 0:
        aux, (grad_coarse, grad_fine) = jax.value_and_grad(
            loss_fn, argnums=[0, 1], has_aux=True
        )(opt_coarse.target, opt_fine.target)

        grad_fine = lax.pmean(grad_fine, axis_name="batch")
        new_opt_fine = opt_fine.apply_gradient(grad_fine, learning_rate=lr)
    else:
        aux, grad_coarse = jax.value_and_grad(loss_fn, has_aux=True)(opt_coarse.target)
        new_opt_fine = None

    grad_coarse = lax.pmean(grad_coarse, axis_name="batch")
    new_opt_coarse = opt_coarse.apply_gradient(grad_coarse, learning_rate=lr)

    new_state = state.replace(
        step=state.step + 1,
        optimizer_coarse=new_opt_coarse,
        optimizer_fine=new_opt_fine,
    )
    loss, (loss_c, loss_f) = aux
    metrics = {
        "loss": loss,
        "loss_c": loss_c,
        "psnr": psnr_fn(loss_f) if config.num_importance > 0 else psnr_fn(loss_c),
        "psnr_c": psnr_fn(loss_c),
        "lr": lr,
    }
    if config.num_importance > 0:
        metrics.update({"loss_f": loss_f, "psnr_f": psnr_fn(loss_f)})
    metrics = lax.pmean(metrics, axis_name="batch")
    return new_state, metrics


def eval_step(model_fn, config, state, rays):
    apply_coarse, apply_fine = model_fn
    opt_coarse, opt_fine = state.optimizer_coarse, state.optimizer_fine

    raw2outputs_ = functools.partial(
        raw2outputs,
        raw_noise_std=0.0,
        white_bkgd=config.white_bkgd,
    )
    *rays, viewdirs = jnp.split(rays, [3, 6, 7, 8], axis=-1)

    pts, z_vals = render_rays(rays, config)
    raw_c = apply_coarse({"params": opt_coarse.target}, pts, viewdirs)
    raw_c = jnp.reshape(raw_c, [-1, config.num_samples, 4])
    coarse_res, weights = raw2outputs_(raw_c, z_vals, rays[1])

    if config.num_importance > 0:
        pts, z_vals, z_std = render_rays_fine(
            rays[:2], z_vals, weights, config.num_importance, perturbation=False
        )
        raw_f = apply_fine({"params": opt_fine.target}, pts, viewdirs)
        raw_f = jnp.reshape(raw_f, [-1, config.num_samples + config.num_importance, 4])
        fine_res, _ = raw2outputs_(raw_f, z_vals, rays[1])
        return fine_res, coarse_res, z_std

    return coarse_res, None, None
