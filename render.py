import functools
import os

import imageio
import jax
import numpy as np
import tensorflow as tf

from absl import app, flags, logging
from flax import jax_utils, optim, struct
from flax.training import checkpoints, common_utils
from jax import numpy as jnp, lax
from jax.config import config as jax_config
from ml_collections import config_flags
from tqdm import tqdm
from typing import Optional

from datasets.input_pipeline import get_dataset
from model import NeRF
from utils import eval_step

jax_config.enable_omnistaging()
tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "File path to the hyperparameter configuration."
)
flags.DEFINE_integer("seed", default=0, help="Initialization seed.")
flags.DEFINE_string("data_dir", default=None, help="Directory containing data files.")
flags.DEFINE_string("model_dir", default=None, help="Directory to store model data.")
flags.DEFINE_string("save_dir", default=None, help="Directory to store outputs.")
flags.DEFINE_string(
    "render_video_set",
    default="render",
    help="Subset of data to use to render the video.",
)
flags.DEFINE_bool("render_video", default=True, help="Whether to render video.")
flags.DEFINE_bool("render_testset", default=False, help="Whether to render testset.")


to_np = (
    lambda x, y, z: np.asarray(x)
    .reshape(1, y[0] + z, y[1], -1)
    .astype(np.float32)[:, z:]
)
to_rgb = lambda x: (255 * np.clip(np.asarray(x), 0, 1)).astype(np.uint8)
psnr_fn = lambda x: -10.0 * np.log(x) / np.log(10.0)


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


def gen_video(data, filename, hwf, step, ch=3):
    img_h, img_w, _ = hwf
    data = to_rgb(data.reshape([-1, img_h, img_w, ch]))
    if FLAGS.save_dir is None:
        out_path = FLAGS.model_dir
    else:
        out_path = FLAGS.save_dir
    imageio.mimwrite(os.path.join(out_path, f"{filename}_{step:06d}.mp4"), data, fps=30)
    imageio.mimwrite(os.path.join(out_path, f"{filename}_{step:06d}.gif"), data, fps=30)


def save_test_imgs(data, hwf, step, idx, ch=3):
    img_h, img_w, _ = hwf
    data = to_rgb(data.reshape([img_h, img_w, ch]))
    if FLAGS.save_dir is None:
        out_path = FLAGS.model_dir
    else:
        out_path = FLAGS.save_dir
    save_path = os.path.join(out_path, f"testset_{step:06d}")
    os.makedirs(save_path, exist_ok=True)
    imageio.imwrite(os.path.join(save_path, f"{idx:02d}.png"), data)


def initialized(key, pts_shape, viewdirs_shape, model_config):
    model = NeRF(
        **model_config,
        **FLAGS.config.emb,
        use_viewdirs=FLAGS.config.use_viewdirs,
        dtype=FLAGS.config.dtype,
    )
    initial_params = jax.jit(model.init)(
        {"params": key},
        jnp.ones(pts_shape, model.dtype),
        jnp.ones(viewdirs_shape, model.dtype),
    )
    return model, initial_params["params"]


@struct.dataclass
class TrainState:
    step: int
    optimizer: optim.Optimizer


def main(_):
    assert FLAGS.config.down_factor > 0 and FLAGS.config.render_factor > 0
    logging.info("JAX host: %d / %d", jax.host_id(), jax.host_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, rng_coarse, rng_fine = jax.random.split(rng, 3)

    ### Load dataset and data values
    datasets, counts, optics, render_datasets = get_dataset(
        FLAGS.data_dir, FLAGS.config, num_poses=FLAGS.config.num_poses
    )
    train_ds, val_ds, test_ds = datasets
    train_items, val_items, test_items = counts
    hwf, r_hwf, near, far = optics
    render_ds, render_vdirs_ds, num_poses = render_datasets

    logging.info("Num poses: %d", num_poses)
    logging.info("Splits: train - %d, val - %d, test - %d", *counts)
    logging.info("Images: height %d, width %d, focal %.5f", *hwf)
    logging.info("Render: height %d, width %d, focal %.5f", *r_hwf)

    ### Init model parameters and optimizer
    initialized_ = functools.partial(initialized, model_config=FLAGS.config.model)
    pts_shape = (FLAGS.config.num_rand, FLAGS.config.num_samples, 3)
    views_shape = (FLAGS.config.num_rand, 3)
    model_coarse, params_coarse = initialized_(rng_coarse, pts_shape, views_shape)

    optimizer = optim.Adam()
    state = TrainState(step=0, optimizer=optimizer.create({"coarse": params_coarse}))
    model_fn = (model_coarse.apply, None)

    if FLAGS.config.num_importance > 0:
        pts_shape = (
            FLAGS.config.num_rand,
            FLAGS.config.num_importance + FLAGS.config.num_samples,
            3,
        )
        model_fine, params_fine = initialized_(rng_fine, pts_shape, views_shape)
        params_dict = {"coarse": params_coarse, "fine": params_fine}
        state = TrainState(step=0, optimizer=optimizer.create(params_dict))
        model_fn = (model_coarse.apply, model_fine.apply)

    state = checkpoints.restore_checkpoint(FLAGS.model_dir, state)
    step = int(state.step)
    state = jax_utils.replicate(state)

    # TODO: TPU Colab breaks without message if this is a list
    # a list is preferred bc tqdm can show an ETA
    render_dict = {
        "train": zip(range(train_items), train_ds),
        "val": zip(range(val_items), val_ds),
        "test": zip(range(test_items), test_ds),
        "poses": zip(range(num_poses), render_ds),
    }
    render_poses = render_dict[FLAGS.render_video_set]

    def render_fn(state, rays):
        step_fn = functools.partial(eval_step, model_fn, FLAGS.config, near, far, state)
        return lax.map(step_fn, rays)

    p_eval_step = jax.pmap(
        render_fn,
        axis_name="batch",
        # in_axes=(0, 0, None),
        # donate_argnums=(0, 1))
    )

    if FLAGS.render_video:
        rgb_list = []
        disp_list = []
        losses = []
        for _, inputs in tqdm(render_poses, desc="Rays render"):
            rays, padding = prepare_render_data(inputs["rays"]._numpy())
            preds, *_ = p_eval_step(state, rays)
            preds = jax.tree_map(lambda x: to_np(x, r_hwf, padding), preds)
            rgb_list.append(preds["rgb"])
            disp_list.append(preds["disp"])

            if FLAGS.config.render_factor == 1 and FLAGS.render_video_set != "render":
                loss = np.mean((preds["rgb"] - inputs["image"]) ** 2.0)
                losses.append(loss)

        if FLAGS.config.render_factor == 1 and FLAGS.render_video_set != "render":
            loss = np.mean(losses)
            logging.info("Loss %.5f", loss)
            logging.info("PSNR %.5f", psnr_fn(loss))
        gen_video(np.stack(rgb_list), "rgb", r_hwf, step)
        disp = np.stack(disp_list)
        gen_video(disp_post(disp, FLAGS.config), "disp", r_hwf, step, ch=1)

    if FLAGS.render_testset:
        test_losses = []
        for idx, inputs in tqdm(zip(range(test_items), test_ds), desc="Test render"):
            rays, padding = prepare_render_data(inputs["rays"]._numpy())
            preds, *_ = p_eval_step(state, rays)
            preds = jax.tree_map(lambda x: to_np(x, r_hwf, padding), preds)
            save_test_imgs(preds["rgb"], r_hwf, step, idx)

            if FLAGS.config.render_factor == 1:
                loss = np.mean((preds["rgb"] - inputs["image"]) ** 2.0)
                test_losses.append(loss)
        if FLAGS.config.render_factor == 1:
            loss = np.mean(test_losses)
            logging.info("Loss %.5f", loss)
            logging.info("PSNR %.5f", psnr_fn(loss))


if __name__ == "__main__":
    flags.mark_flags_as_required(["data_dir", "config", "model_dir"])
    logging.set_verbosity(logging.INFO)
    app.run(main)
