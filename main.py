import functools
import os

import jax
import numpy as np
import optax
import tensorflow as tf

from absl import app, flags, logging
from clu import metric_writers, parameter_overview, periodic_actions, platform
from flax.training import checkpoints, common_utils, train_state
from jax import numpy as jnp, lax
from ml_collections import config_flags
from tqdm import tqdm

from datasets.input_pipeline import get_dataset
from model import NeRF
from utils import (
    disp_post,
    eval_step,
    gen_video,
    prepare_render_data,
    save_checkpoint,
    save_test_imgs,
    to_rgb,
    to_np,
    train_step,
)

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "File path to the hyperparameter configuration."
)
flags.DEFINE_integer("seed", default=0, help="Initialization seed.")
flags.DEFINE_string("data_dir", default=None, help="Directory containing data files.")
flags.DEFINE_string("model_dir", default=None, help="Directory to store model data.")

psnr_fn = lambda x: -10.0 * np.log(x) / np.log(10.0)


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


def main(_):
    if FLAGS.config.precrop_iters > 0 and FLAGS.config.batching:
        raise ValueError("'precrop_iters has no effect when 'batching' the dataset")
    assert FLAGS.config.down_factor > 0 and FLAGS.config.render_factor > 0

    logging.info("JAX host: %d / %d", jax.process_index(), jax.host_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    platform.work_unit().set_task_status(
        f"host_id: {jax.process_index()}, host_count: {jax.host_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.model_dir, "model_dir"
    )

    os.makedirs(FLAGS.model_dir, exist_ok=True)
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, rng_coarse, rng_fine, data_rng, step_rng = jax.random.split(rng, 5)
    rngs = common_utils.shard_prng_key(step_rng)

    ### Load dataset and data values
    datasets, counts, optics, render_datasets = get_dataset(
        FLAGS.data_dir, FLAGS.config, rng=data_rng, num_poses=FLAGS.config.num_poses
    )
    train_ds, val_ds, test_ds = datasets
    *_, test_items = counts
    hwf, r_hwf, near, far = optics
    render_ds, render_vdirs_ds, num_poses = render_datasets
    iter_render_ds = zip(range(num_poses), render_ds)
    iter_vdirs_ds = zip(range(num_poses), render_vdirs_ds)
    iter_test_ds = zip(range(test_items), test_ds)
    img_h, img_w, _ = hwf

    logging.info("Num poses: %d", num_poses)
    logging.info("Splits: train - %d, val - %d, test - %d", *counts)
    logging.info("Images: height %d, width %d, focal %.5f", *hwf)
    logging.info("Render: height %d, width %d, focal %.5f", *r_hwf)

    ### Init model parameters and optimizer
    initialized_ = functools.partial(initialized, model_config=FLAGS.config.model)
    pts_shape = (FLAGS.config.num_rand, FLAGS.config.num_samples, 3)
    views_shape = (FLAGS.config.num_rand, 3)
    model_coarse, params_coarse = initialized_(rng_coarse, pts_shape, views_shape)

    schedule_fn = optax.exponential_decay(
        init_value=FLAGS.config.learning_rate,
        transition_steps=FLAGS.config.lr_decay * 1000,
        decay_rate=FLAGS.config.decay_factor,
    )
    tx = optax.adam(learning_rate=schedule_fn)
    state = train_state.TrainState.create(
        apply_fn=(model_coarse.apply, None), params={"coarse": params_coarse}, tx=tx
    )

    if FLAGS.config.num_importance > 0:
        pts_shape = (
            FLAGS.config.num_rand,
            FLAGS.config.num_importance + FLAGS.config.num_samples,
            3,
        )
        model_fine, params_fine = initialized_(rng_fine, pts_shape, views_shape)
        state = train_state.TrainState.create(
            apply_fn=(model_coarse.apply, model_fine.apply),
            params={"coarse": params_coarse, "fine": params_fine},
            tx=tx,
        )

    state = checkpoints.restore_checkpoint(FLAGS.model_dir, state)
    start_step = int(state.step)

    # cycle already seen examples if resuming from checkpoint
    # (only useful for ensuring deterministic dataset, slow for large start_step)
    # if start_step != 0:
    #     for _ in range(start_step):
    #         _ = next(train_ds)

    # parameter_overview.log_parameter_overview(state.optimizer_coarse.target)
    # if FLAGS.config.num_importance > 0:
    #     parameter_overview.log_parameter_overview(state.optimizer_fine.target)

    state = jax.device_put_replicated(state, jax.local_devices())

    ### Build "pmapped" functions for distributed training
    train_fn = functools.partial(train_step, near, far, FLAGS.config, schedule_fn)
    p_train_step = jax.pmap(
        train_fn,
        axis_name="batch",
        in_axes=(0, 0, None, 0),
        # donate_argnums=(0, 1, 2),
    )

    def render_fn(state, rays):
        step_fn = functools.partial(eval_step, FLAGS.config, near, far, state)
        return lax.map(step_fn, rays)

    p_eval_step = jax.pmap(
        render_fn,
        axis_name="batch",
        # in_axes=(0, 0, None),
        # donate_argnums=(0, 1))
    )

    # TODO: add hparams
    writer = metric_writers.create_default_writer(
        FLAGS.model_dir, just_logging=jax.process_index() > 0
    )
    logging.info("Starting training loop.")

    hooks = []
    profiler = periodic_actions.Profile(num_profile_steps=5, logdir=FLAGS.model_dir)
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=FLAGS.config.num_steps, writer=writer
    )
    if jax.process_index() == 0:
        hooks += [profiler, report_progress]
    train_metrics = []
    gen_video_ = functools.partial(gen_video, FLAGS.model_dir)

    for step in range(start_step, FLAGS.config.num_steps + 1):
        is_last_step = step == FLAGS.config.num_steps

        batch = next(train_ds)
        coords = None
        if not FLAGS.config.batching:
            coords = jnp.meshgrid(jnp.arange(img_h), jnp.arange(img_w), indexing="ij")
            if step < FLAGS.config.precrop_iters:
                dH = int(img_h // 2 * FLAGS.config.precrop_frac)
                dW = int(img_w // 2 * FLAGS.config.precrop_frac)
                coords = jnp.meshgrid(
                    jnp.arange(img_h // 2 - dH, img_h // 2 + dH),
                    jnp.arange(img_w // 2 - dW, img_w // 2 + dW),
                    indexing="ij",
                )
            coords = jnp.stack(coords, axis=-1).reshape([-1, 2])

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            state, metrics = p_train_step(batch, state, coords, rngs)
        train_metrics.append(metrics)

        logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
        _ = [h(step) for h in hooks]

        ### Write train summaries to TB
        if step % FLAGS.config.i_print == 0 or is_last_step:
            with report_progress.timed("training_metrics"):
                train_metrics = common_utils.get_metrics(train_metrics)
                train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
                summary = {f"train/{k}": v for k, v in train_summary.items()}
                writer.write_scalars(step, summary)
            train_metrics = []

        ### Eval a random validation image and plot it to TB
        if step % FLAGS.config.i_img == 0 and step > 0 or is_last_step:
            with report_progress.timed("validation"):
                inputs = next(val_ds)
                rays, padding = prepare_render_data(inputs["rays"]._numpy())
                outputs = p_eval_step(state, rays)
                preds, preds_c, z_std = jax.tree_map(
                    lambda x: to_np(x, hwf, padding), outputs
                )
                loss = np.mean((preds["rgb"] - inputs["image"]) ** 2)
                summary = {"val/loss": loss, "val/psnr": psnr_fn(loss)}
                writer.write_scalars(step, summary)

                summary = {
                    "val/rgb": to_rgb(preds["rgb"]),
                    "val/target": to_np(inputs["image"], hwf, padding),
                    "val/disp": disp_post(preds["disp"], FLAGS.config),
                    "val/acc": preds["acc"],
                }
                if FLAGS.config.num_importance > 0:
                    summary["val/rgb_c"] = to_rgb(preds_c["rgb"])
                    summary["val/disp_c"] = disp_post(preds_c["disp"], FLAGS.config)
                    summary["val/z_std"] = z_std
                writer.write_images(step, summary)

        ### Render a video with test poses
        if step % FLAGS.config.i_video == 0 and step > 0:
            with report_progress.timed("video_render"):
                logging.info("Rendering video at step %d", step)
                rgb_list = []
                disp_list = []
                for idx, inputs in tqdm(iter_render_ds, desc="Rays render"):
                    rays, padding = prepare_render_data(inputs["rays"].numpy())
                    preds, *_ = p_eval_step(state, rays)
                    preds = jax.tree_map(lambda x: to_np(x, r_hwf, padding), preds)
                    rgb_list.append(preds["rgb"])
                    disp_list.append(preds["disp"])

                gen_video_(np.stack(rgb_list), "rgb", r_hwf, step)
                disp = np.stack(disp_list)
                gen_video_(disp_post(disp, FLAGS.config), "disp", r_hwf, step, ch=1)

                if FLAGS.config.use_viewdirs:
                    rgb_list = []
                    for idx, inputs in tqdm(iter_vdirs_ds, desc="Viewdirs render"):
                        rays, padding = prepare_render_data(inputs["rays"].numpy())
                        preds, *_ = p_eval_step(state, rays)
                        rgb_list.append(to_np(preds["rgb"], r_hwf, padding))
                    gen_video_(np.stack(rgb_list), "rgb_still", r_hwf, step)

        ### Save images in the test set
        if step % FLAGS.config.i_testset == 0 and step > 0:
            with report_progress.timed("test_render"):
                logging.info("Rendering test set at step %d", step)
                test_losses = []
                for idx, inputs in tqdm(iter_test_ds, desc="Test render"):
                    rays, padding = prepare_render_data(inputs["rays"].numpy())
                    preds, *_ = p_eval_step(state, rays)
                    save_test_imgs(FLAGS.model_dir, preds["rgb"], r_hwf, step, idx)

                    if FLAGS.config.render_factor == 0:
                        loss = np.mean((preds["rgb"] - inputs["image"]) ** 2.0)
                        test_losses.append(loss)
                if FLAGS.config.render_factor == 0:
                    loss = np.mean(test_losses)
                    summary = {"test/loss": loss, "test/psnr": psnr_fn(loss)}
                    writer.write_scalars(step, summary)
        writer.flush()

        ### Save ckpt
        if step % FLAGS.config.i_weights == 0 or is_last_step:
            with report_progress.timed("checkpoint"):
                save_checkpoint(state, FLAGS.model_dir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["data_dir", "config", "model_dir"])
    logging.set_verbosity(logging.INFO)
    app.run(main)
