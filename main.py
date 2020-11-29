import functools
import os
import time

import imageio
import jax
import numpy as np

from absl import app, flags, logging
from flax import jax_utils, optim, struct
from flax.metrics import tensorboard
from flax.training import checkpoints, common_utils
from jax import numpy as jnp, lax, random
from jax.config import config as jax_config
from ml_collections import config_flags
from typing import Any, Optional

from datasets import load_blender, load_deepvoxels
from model import NeRF
from rays_utils import prepare_rays
from utils import create_learning_rate_scheduler, eval_step, train_step

jax_config.enable_omnistaging()

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    os.path.join(os.path.dirname(__file__), "configs/default.py"),
    "File path to the hyperparameter configuration.",
)

flags.DEFINE_integer("seed", default=0, help=("Initialization seed."))
flags.DEFINE_string("data_dir", default=None, help=("Directory containing data files."))
flags.DEFINE_string("model_dir", default=None, help=("Directory to store model data."))
flags.DEFINE_string(
    "jax_backend_target",
    default=None,
    help=("JAX backend target to use. Can be used with UPTC."),
)


def gen_video(data, filename, hwf, step, ch=3):
    img_h, img_w, _ = hwf
    data = 255 * np.clip(data.reshape([-1, img_h, img_w, ch]), 0, 1)
    imageio.mimwrite(
        os.path.join(FLAGS.model_dir, f"{filename}_{step:06d}.mp4"),
        data.astype(np.uint8),
        fps=30,
        quality=8,
    )


def save_test_imgs(data, hwf, step, ch=3):
    img_h, img_w, _ = hwf
    data = 255 * np.clip(data.reshape([-1, img_h, img_w, ch]), 0, 1)
    save_path = os.path.join(FLAGS.model_dir, f"testset_{step:06d}")
    os.makedirs(save_path, exist_ok=True)
    [
        imageio.imwrite(os.path.join(save_path, f"{idx:02d}.png"), x)
        for idx, x in enumerate(data.astype(np.uint8))
    ]


def initialized(key, input_pts_shape, input_viewdirs_shape, model_config):
    model = NeRF(
        **model_config,
        **FLAGS.config.emb,
        use_viewdirs=FLAGS.config.use_viewdirs,
        dtype=FLAGS.config.dtype,
    )
    initial_params = model.init(
        {"params": key},
        jnp.ones(input_pts_shape, model.dtype),
        jnp.ones(input_viewdirs_shape, model.dtype),
    )
    return model, initial_params["params"]


@struct.dataclass
class TrainState:
    step: int
    optimizer_coarse: optim.Optimizer
    optimizer_fine: Optional[optim.Optimizer]


def main(_):
    if FLAGS.jax_backend_target:
        logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
        jax_config.update("jax_xla_backend", "tpu_driver")
        jax_config.update("jax_backend_target", FLAGS.jax_backend_target)

    logging.info("JAX host: %d / %d", jax.host_id(), jax.host_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    if jax.host_id() == 0:
        summary_writer = tensorboard.SummaryWriter(FLAGS.model_dir)
        # summary_writer.hparams(dict(FLAGS.config))

    rng = random.PRNGKey(FLAGS.seed)
    rng, init_rng_coarse, init_rng_fine = random.split(rng, 3)
    n_devices = jax.device_count()

    ### Load dataset and data values
    if FLAGS.config.dataset_type == "blender":
        images, poses, render_poses, hwf, counts = load_blender.load_data(
            FLAGS.data_dir,
            half_res=FLAGS.config.half_res,
            testskip=FLAGS.config.testskip,
        )
        logging.info("Loaded blender, total images: %d", images.shape[0])

        near = 2.0
        far = 6.0

        if FLAGS.config.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif FLAGS.config.dataset_type == "deepvoxels":
        images, poses, render_poses, hwf, counts = load_deepvoxels.load_dv_data(
            FLAGS.data_dir,
            scene=FLAGS.config.shape,
            testskip=FLAGS.config.testskip,
        )
        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.0
        far = hemi_R + 1.0
        logging.info(
            "Loaded deepvoxels (%s), total images: %d",
            FLAGS.config.shape,
            images.shape[0],
        )
    else:
        raise ValueError(f"Dataset '{FLAGS.config.dataset_type}' is not available.")

    img_h, img_w, focal = hwf
    logging.info("Images splits: %s", counts)
    logging.info("Render poses: %s", render_poses.shape)
    logging.info("Image height: %d, image width: %d, focal: %.5f", img_h, img_w, focal)

    train_imgs, val_imgs, test_imgs, *_ = np.split(images, np.cumsum(counts))
    train_poses, val_poses, test_poses, *_ = np.split(poses, np.cumsum(counts))

    if FLAGS.config.render_factor > 0:
        # render downsampled for speed
        r_img_h = img_h // FLAGS.config.render_factor
        r_img_w = img_w // FLAGS.config.render_factor
        r_focal = focal / FLAGS.config.render_factor
        r_hwf = r_img_h, r_img_w, r_focal
    else:
        r_hwf = hwf

    to_np = lambda x, h=img_h, w=img_w: np.reshape(x, [h, w, -1]).astype(np.float32)
    psnr_fn = lambda x: -10.0 * np.log(x) / np.log(10.0)

    ### Pre-compute rays
    @functools.partial(jax.jit, static_argnums=(0,))
    def prep_rays(hwf, c2w, c2w_sc=None):
        if c2w_sc is not None:
            c2w_sc = c2w_sc[:3, :4]
        return prepare_rays(None, hwf, FLAGS.config, near, far, c2w[:3, :4], c2w_sc)

    rays_render = lax.map(lambda x: prep_rays(r_hwf, x), render_poses)
    render_shape = [-1, n_devices, r_hwf[1], rays_render.shape[-1]]
    rays_render = jnp.reshape(rays_render, render_shape)
    logging.info("Render rays shape: %s", rays_render.shape)

    if FLAGS.config.use_viewdirs:
        rays_render_vdirs = lax.map(
            lambda x: prep_rays(r_hwf, x, render_poses[0]), render_poses
        ).reshape(render_shape)

    if FLAGS.config.batching:
        train_rays = lax.map(lambda pose: prep_rays(hwf, pose), train_poses)
        train_rays = jnp.reshape(train_rays, [-1, train_rays.shape[-1]])
        train_imgs = jnp.reshape(train_imgs, [-1, 3])
        logging.info("Batched rays shape: %s", train_rays.shape)
        val_rays = lax.map(lambda pose: prep_rays(hwf, pose), val_poses)

    test_rays = lax.map(lambda pose: prep_rays(r_hwf, pose), test_poses)
    test_rays = jnp.reshape(test_rays, render_shape)

    ### Init model parameters and optimizer
    input_pts_shape = (FLAGS.config.num_rand, FLAGS.config.num_samples, 3)
    input_views_shape = (FLAGS.config.num_rand, 3)
    model_coarse, params_coarse = initialized(
        init_rng_coarse, input_pts_shape, input_views_shape, FLAGS.config.model
    )

    optimizer = optim.Adam()
    state = TrainState(
        step=0, optimizer_coarse=optimizer.create(params_coarse), optimizer_fine=None
    )
    model_fn = (model_coarse.apply, None)

    if FLAGS.config.num_importance > 0:
        input_pts_shape = (
            FLAGS.config.num_rand,
            FLAGS.config.num_importance + FLAGS.config.num_samples,
            3,
        )
        model_fine, params_fine = initialized(
            init_rng_fine, input_pts_shape, input_views_shape, FLAGS.config.model_fine
        )
        state = state.replace(optimizer_fine=optimizer.create(params_fine))
        model_fn = (model_coarse.apply, model_fine.apply)

    state = checkpoints.restore_checkpoint(FLAGS.model_dir, state)
    start_step = int(state.step)
    state = jax_utils.replicate(state)

    ### Build 'pmapped' functions for distributed training
    learning_rate_fn = create_learning_rate_scheduler(
        factors=FLAGS.config.lr_schedule,
        base_learning_rate=FLAGS.config.learning_rate,
        decay_factor=FLAGS.config.decay_factor,
        steps_per_decay=FLAGS.config.lr_decay * 1000,
    )
    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            model_fn,
            FLAGS.config,
            learning_rate_fn,
            (hwf, near, far),
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )
    p_eval_step = jax.pmap(
        functools.partial(eval_step, model_fn, FLAGS.config),
        axis_name="batch",
    )

    t = time.time()
    train_metrics = []

    for step in range(start_step, FLAGS.config.num_steps + 1):
        rng, sample_rng, step_rng, test_rng = random.split(rng, 4)
        sharded_rngs = common_utils.shard_prng_key(step_rng)
        coords = None

        if FLAGS.config.batching:
            select_idx = random.randint(
                sample_rng,
                [n_devices * FLAGS.config.num_rand],
                minval=0,
                maxval=train_rays.shape[0],
            )
            inputs = train_rays[select_idx, ...]
            inputs = jnp.reshape(inputs, [n_devices, FLAGS.config.num_rand, -1])
            target = train_imgs[select_idx, ...]
            target = jnp.reshape(target, [n_devices, FLAGS.config.num_rand, 3])
        else:
            img_idx = random.randint(
                sample_rng, [n_devices], minval=0, maxval=counts[0]
            )
            inputs = train_poses[img_idx, ...]  # [n_devices, 4, 4]
            target = train_imgs[img_idx, ...]  # [n_devices, img_h, img_w, 3]

            if step < FLAGS.config.precrop_iters:
                dH = int(img_h // 2 * FLAGS.config.precrop_frac)
                dW = int(img_w // 2 * FLAGS.config.precrop_frac)
                coords = jnp.meshgrid(
                    jnp.arange(img_h // 2 - dH, img_h // 2 + dH),
                    jnp.arange(img_w // 2 - dW, img_w // 2 + dW),
                    indexing="ij",
                )
                coords = jax_utils.replicate(
                    jnp.stack(coords, axis=-1).reshape([-1, 2])
                )

        state, metrics, coarse_res, fine_res = p_train_step(
            state, (inputs, target), coords, rng=sharded_rngs
        )
        train_metrics.append(metrics)

        ### Write summaries to TB
        if step % FLAGS.config.i_print == 0 and step > 0:
            steps_per_sec = time.time() - t
            train_metrics = common_utils.get_metrics(train_metrics)
            train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
            if jax.host_id() == 0:
                logging.info(
                    "Step: %6d, %.3f s/step, loss %.5f, psnr %6.3f",
                    step,
                    steps_per_sec,
                    train_summary["loss"],
                    train_summary["psnr"],
                )
                for key, val in train_summary.items():
                    summary_writer.scalar(f"train/{key}", val, step)

                summary_writer.scalar("steps per second", steps_per_sec, step)
                summary_writer.histogram("raw_c", np.array(coarse_res["raw"]), step)
                if FLAGS.config.num_importance > 0:
                    summary_writer.histogram("raw_f", np.array(fine_res["raw"]), step)
            train_metrics = []

            ### Eval a random validation image and plot it in TB
            if step % FLAGS.config.i_img == 0:
                val_idx = random.randint(test_rng, [1], minval=0, maxval=counts[1])
                if FLAGS.config.batching:
                    inputs = val_rays[tuple(val_idx)].reshape(render_shape)
                else:
                    inputs = prep_rays(hwf, val_poses[tuple(val_idx)])
                    inputs = jnp.reshape(inputs, render_shape)
                target = val_imgs[tuple(val_idx)]
                preds, preds_c, z_std = lax.map(lambda x: p_eval_step(state, x), inputs)
                rgb = to_np(preds["rgb"])
                loss = np.mean((rgb - target) ** 2)

                summary_writer.scalar(f"val/loss", loss, step)
                summary_writer.scalar(f"val/psnr", psnr_fn(loss), step)

                rgb = 255 * np.clip(rgb, 0, 1)
                summary_writer.image("val/rgb", rgb.astype(np.uint8), step)
                summary_writer.image("val/target", target, step)
                summary_writer.image("val/disp", to_np(preds["disp"]), step)
                summary_writer.image("val/acc", to_np(preds["acc"]), step)

                if FLAGS.config.num_importance > 0:
                    rgb = 255 * np.clip(to_np(preds_c["rgb"]), 0, 1)
                    summary_writer.image("val/rgb_c", rgb.astype(np.uint8), step)
                    summary_writer.image("val/disp_c", to_np(preds_c["disp"]), step)
                    summary_writer.image("val/z_std", to_np(z_std), step)

        ### Render a video with test poses
        if step % FLAGS.config.i_video == 0 and step > 0:
            logging.info("Rendering video at step %d", step)
            t = time.time()
            preds, *_ = lax.map(lambda x: p_eval_step(state, x), rays_render)
            gen_video(preds["rgb"], "rgb", r_hwf, step)
            gen_video(preds["disp"] / jnp.max(preds["disp"]), "disp", r_hwf, step, ch=1)

            if FLAGS.config.use_viewdirs:
                preds = lax.map(
                    lambda x: p_eval_step(state, x)[0]["rgb"], rays_render_vdirs
                )
                gen_video(preds, "rgb_still", r_hwf, step)
            logging.info("Video rendering done in %ds", time.time() - t)

        ### Save images in the test set
        if step % FLAGS.config.i_testset == 0 and step > 0:
            logging.info("Rendering test set at step %d", step)
            preds = lax.map(lambda x: p_eval_step(state, x)[0]["rgb"], test_rays)
            save_test_imgs(preds, r_hwf, step)

            if FLAGS.config.render_factor == 0:
                loss = np.mean((preds.reshape(test_imgs.shape) - test_imgs) ** 2.0)
                summary_writer.scalar(f"test/loss", loss, step)
                summary_writer.scalar(f"test/psnr", psnr_fn(loss), step)

        ### Save ckpt
        if step % FLAGS.config.i_weights == 0 and step > 0:
            if jax.host_id() == 0:
                checkpoints.save_checkpoint(
                    FLAGS.model_dir,
                    jax_utils.unreplicate(state),
                    step,
                    keep=5,
                )
        t = time.time()


if __name__ == "__main__":
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("data_dir")
    logging.set_verbosity(logging.INFO)
    app.run(main)
