import functools
import os

import imageio
import jax
import numpy as np

from absl import logging
from flax.training import checkpoints
from jax import numpy as jnp, lax, random

from rays_utils import raw2outputs, render_rays, render_rays_fine

psnr_fn = lambda x: -10.0 * jnp.log(x) / jnp.log(10.0)
to_rgb = lambda x: (255 * np.clip(np.asarray(x), 0, 1)).astype(np.uint8)
to_np = (
    lambda x, y, z: np.asarray(x)
    .reshape(1, y[0] + z, y[1], -1)
    .astype(np.float32)[:, z:]
)


def save_checkpoint(state, workdir, keep=3):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=keep)


def disp_post(disp, config):
    if config.dataset_type == "llff":
        disp = (disp - disp.min()) / np.clip(disp.max() - disp.min(), 1e-5, None)
    return disp


def prepare_render_data(rays):
    padding = 0
    img_h, img_w, chn = rays.shape
    rays_remaining = np.prod(img_h) % jax.local_device_count()
    if rays_remaining != 0:
        padding = jax.local_device_count() - rays_remaining
        rays = np.pad(rays, ((padding, 0), (0, 0), (0, 0)), mode="edge")
    return rays.reshape(jax.local_device_count(), -1, img_w, chn), padding


def gen_video(save_dir, data, filename, hwf, step, ch=3):
    img_h, img_w, _ = hwf
    data = to_rgb(data.reshape([-1, img_h, img_w, ch]))
    imageio.mimwrite(os.path.join(save_dir, f"{filename}_{step:06d}.mp4"), data, fps=30)
    # imageio.mimwrite(os.path.join(out_path, f"{filename}_{step:06d}.gif"), data, fps=30)


def save_test_imgs(save_dir, data, hwf, step, idx, ch=3):
    img_h, img_w, _ = hwf
    data = to_rgb(data.reshape([img_h, img_w, ch]))
    save_path = os.path.join(save_dir, f"testset_{step:06d}")
    os.makedirs(save_path, exist_ok=True)
    imageio.imwrite(os.path.join(save_path, f"{idx:02d}.png"), data)


def train_step(
    near,
    far,
    config,
    lr_fn,
    batch,
    state,
    coords=None,
    rng=None,
):
    """Perform a single training step."""
    rng = jax.random.fold_in(rng, state.step)
    rng0, rng1, rng2, rng3, rng4 = random.split(rng, 5)
    inputs, targets = batch["rays"], batch["image"]
    model_coarse, model_fine = state.apply_fn
    step = state.step

    if not config.batching:
        select_idx = random.choice(
            rng0,
            coords.shape[0],
            shape=[config.num_rand],
            replace=False,
        )
        select_idx = coords[select_idx]
        rays = inputs[select_idx[:, 0], select_idx[:, 1]]
        targets = targets[select_idx[:, 0], select_idx[:, 1]]
    else:
        rays = inputs

    *rays, viewdirs = jnp.split(rays, [3, 6], axis=-1)
    _, rays_d = rays

    render_rays_fine_ = functools.partial(
        render_rays_fine,
        num_importance=config.num_importance,
        perturbation=config.perturb,
    )
    raw2outputs_ = functools.partial(
        raw2outputs,
        raw_noise_std=config.raw_noise_std,
        white_bkgd=config.white_bkgd,
    )

    def loss_fn(params):
        """Loss function used for training."""
        pts, z_vals = render_rays(*rays, config, near, far, rng1)
        raw_c = model_coarse({"params": params["coarse"]}, pts, viewdirs).reshape(
            [config.num_rand, config.num_samples, 4]
        )
        coarse_res, weights = raw2outputs_(raw_c, z_vals, rays_d, rng=rng2)
        loss_c = jnp.mean((coarse_res["rgb"] - targets) ** 2.0)

        loss_f = 0
        if config.num_importance > 0:
            pts, z_vals, _ = render_rays_fine_(*rays, z_vals, weights, rng=rng3)
            raw_f = model_fine({"params": params["fine"]}, pts, viewdirs).reshape(
                [config.num_rand, config.num_samples + config.num_importance, 4]
            )
            fine_res, _ = raw2outputs_(raw_f, z_vals, rays_d, rng=rng4)
            loss_f = jnp.mean((fine_res["rgb"] - targets) ** 2.0)

        loss = loss_c + loss_f
        return loss, (loss_c, loss_f)

    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    loss, (loss_c, loss_f) = aux
    metrics = {
        "loss": loss,
        "loss_c": loss_c,
        "psnr": psnr_fn(loss_f) if config.num_importance > 0 else psnr_fn(loss_c),
        "psnr_c": psnr_fn(loss_c),
        "lr": lr_fn(step),
    }
    if config.num_importance > 0:
        metrics.update({"loss_f": loss_f, "psnr_f": psnr_fn(loss_f)})
    metrics = lax.pmean(metrics, axis_name="batch")
    return new_state, metrics


def eval_step(config, near, far, state, rays):
    apply_coarse, apply_fine = state.apply_fn
    render_rays_fine_ = functools.partial(
        render_rays_fine,
        num_importance=config.num_importance,
        perturbation=False,
    )
    raw2outputs_ = functools.partial(
        raw2outputs,
        raw_noise_std=0.0,
        white_bkgd=config.white_bkgd,
    )
    rays_o, rays_d, viewdirs = jnp.split(rays, [3, 6], axis=-1)

    pts, z_vals = render_rays(rays_o, rays_d, config, near, far)
    raw_c = apply_coarse({"params": state.params["coarse"]}, pts, viewdirs)
    raw_c = jnp.reshape(raw_c, [-1, config.num_samples, 4])
    coarse_res, weights = raw2outputs_(raw_c, z_vals, rays_d)

    if config.num_importance > 0:
        pts, z_vals, z_std = render_rays_fine_(rays_o, rays_d, z_vals, weights)
        raw_f = apply_fine({"params": state.params["fine"]}, pts, viewdirs)
        raw_f = jnp.reshape(raw_f, [-1, config.num_samples + config.num_importance, 4])
        fine_res, _ = raw2outputs_(raw_f, z_vals, rays_d)
        return fine_res, coarse_res, z_std

    return coarse_res, None, None
