import functools
import os
import time

import imageio
import jax
import numpy as np

from absl import app, flags, logging
from flax import jax_utils, optim, struct
from flax.training import checkpoints
from jax import numpy as jnp, lax, random
from jax.config import config as jax_config
from ml_collections import config_flags
from typing import Any, Optional

from datasets import load_blender, load_deepvoxels
from model import NeRF
from rays_utils import prepare_rays
from utils import eval_step, psnr_fn

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
flags.DEFINE_string("save_dir", default=None, help=("Directory to store outputs."))
flags.DEFINE_string(
    "render_video_set",
    default="render",
    help=("Subset of data to use to render the video."),
)
flags.DEFINE_bool("render_video", default=True, help=("Whether to render video."))
flags.DEFINE_bool("render_testset", default=True, help=("Whether to render testset."))
flags.DEFINE_string(
    "jax_backend_target",
    default=None,
    help=("JAX backend target to use. Can be used with UPTC."),
)


def gen_video(data, filename, hwf, step, ch=3):
    img_h, img_w, _ = hwf
    data = 255 * jnp.clip(data.reshape([-1, img_h, img_w, ch]), 0, 1)
    if FLAGS.save_dir is None:
        out_path = FLAGS.model_dir
    else:
        out_path = FLAGS.save_dir
    imageio.mimwrite(
        os.path.join(out_path, f"{filename}_{step:06d}.mp4"),
        data.astype(jnp.uint8),
        fps=30,
        quality=8,
    )
    imageio.mimwrite(
        os.path.join(out_path, f"{filename}_{step:06d}.gif"),
        data.astype(jnp.uint8),
        fps=30,
    )


def save_test_imgs(data, hwf, step, ch=3):
    img_h, img_w, _ = hwf
    data = 255 * jnp.clip(data.reshape([-1, img_h, img_w, ch]), 0, 1)
    if FLAGS.save_dir is None:
        out_path = FLAGS.model_dir
    else:
        out_path = FLAGS.save_dir
    save_path = os.path.join(out_path, f"testset_{step:06d}")
    os.makedirs(save_path, exist_ok=True)
    [
        imageio.imwrite(os.path.join(save_path, f"{idx:02d}.png"), x)
        for idx, x in enumerate(data.astype(jnp.uint8))
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

    ### Pre-compute rays
    @functools.partial(jax.jit, static_argnums=(0,))
    def prep_rays(hwf, c2w, c2w_sc=None):
        if c2w_sc is not None:
            c2w_sc = c2w_sc[:3, :4]
        return prepare_rays(None, hwf, FLAGS.config, near, far, c2w[:3, :4], c2w_sc)

    render_dict = {
        "train": train_poses,
        "val": val_poses,
        "test": test_poses,
        "render": render_poses,
    }
    render_poses = render_dict[FLAGS.render_video_set]

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
    state_step = int(state.step)
    state = jax_utils.replicate(state)

    p_eval_step = jax.pmap(
        functools.partial(eval_step, model_fn, FLAGS.config),
        axis_name="batch",
    )
    eval_fn = lambda x: p_eval_step(state, x)[0]["rgb"]

    if FLAGS.render_video:
        rays_render = lax.map(lambda x: prep_rays(r_hwf, x), render_poses)
        render_shape = [-1, n_devices, r_hwf[1], rays_render.shape[-1]]
        rays_render = jnp.reshape(rays_render, render_shape)
        logging.info("Render rays shape: %s", rays_render.shape)

        logging.info("Rendering video at step %d", state_step)
        t = time.time()
        preds, *_ = lax.map(lambda x: p_eval_step(state, x), rays_render)
        gen_video(preds["rgb"], "rgb", r_hwf, state_step)
        gen_video(
            preds["disp"] / jnp.max(preds["disp"]), "disp", r_hwf, state_step, ch=1
        )
        if FLAGS.render_video_set == "test" and FLAGS.config.render_factor == 0:
            loss = jnp.mean((preds["rgb"].reshape(test_imgs.shape) - test_imgs) ** 2.0)
            logging.info("test/loss %.5f", loss)
            logging.info("test/psnr %.5f", psnr_fn(loss))

        if FLAGS.config.use_viewdirs:
            logging.info("Rendering video for view directions at step %d", state_step)
            rays_render_vdirs = lax.map(
                lambda x: prep_rays(r_hwf, x, render_poses[0]), render_poses
            ).reshape(render_shape)
            preds = lax.map(eval_fn, rays_render_vdirs)
            gen_video(preds, "rgb_still", r_hwf, state_step)
        logging.info("Video rendering done in %ds", time.time() - t)

    if FLAGS.render_testset:
        test_rays = lax.map(lambda pose: prep_rays(r_hwf, pose), test_poses)
        test_rays = jnp.reshape(
            test_rays, [-1, n_devices, r_hwf[1], test_rays.shape[-1]]
        )
        logging.info("Test rays shape: %s", test_rays.shape)
        logging.info("Rendering test set at step %d", state_step)
        preds = lax.map(eval_fn, test_rays)
        save_test_imgs(preds, r_hwf, state_step)

        if FLAGS.config.render_factor == 0:
            loss = jnp.mean((preds.reshape(test_imgs.shape) - test_imgs) ** 2.0)
            logging.info("test/loss %.5f", loss)
            logging.info("test/psnr %.5f", psnr_fn(loss))


if __name__ == "__main__":
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("data_dir")
    logging.set_verbosity(logging.INFO)
    app.run(main)
