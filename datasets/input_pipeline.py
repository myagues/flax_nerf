import functools

import jax
import tensorflow as tf

from flax import jax_utils

from clu import deterministic_data
from datasets import load_blender, load_deepvoxels, load_llff


AUTOTUNE = tf.data.AUTOTUNE


def prepare_train_data(dataset):
    """Convert a input batch from TF tensors to NumPy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        x = x._numpy()
        return x.reshape((local_device_count,) + x.shape[1:])

    it = map(lambda x: jax.tree_map(_prepare, x), dataset)
    return jax_utils.prefetch_to_device(it, 2)


def batching_sample(features, config):
    rays_range, _ = tf.unstack(tf.shape(features["rays"]))
    kwargs = {"minval": 0, "maxval": rays_range, "dtype": tf.int32}
    if "rng" in features:
        idx = tf.random.stateless_uniform([config.num_rand], features["rng"], **kwargs)
    else:
        idx = tf.random.uniform([config.num_rand], **kwargs)
    rays = tf.gather(features["rays"], idx)
    image = tf.gather(features["image"], idx)
    return {"rays": rays, "image": image}


def prepare_rays(features, config, c2w_static_cam=None):
    """
    Build rays for rendering.
    Args:
        rays: (2, num_rays, 3) origin and direction generated rays
        hwf: (3) tuple containing image height, width and focal length
        config: model and rendering config
        c2w: (3, 4) camera-to-world transformation matrix
        c2w_static_cam: (3, 4) transformation matrix for camera
    Returns:
        rays: (img_h, img_w, *) generated rays
    """
    rays_o, rays_d = get_rays(*features["hwf"], features["pose"])

    viewdirs = None
    if config.use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        # make all directions unit magnitude
        viewdirs /= tf.linalg.norm(viewdirs, axis=-1, keepdims=True)

        if c2w_static_cam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(*features["hwf"], c2w_static_cam)

    # for forward facing scenes
    if not config.llff.spherify and config.dataset_type == "llff":
        rays_o, rays_d = ndc_rays(*features["hwf"], 1.0, rays_o, rays_d)

    rays = [rays_o, rays_d]
    if viewdirs is not None:
        rays.append(viewdirs)
    rays = tf.concat(rays, axis=-1)
    return {"rays": rays, "image": features["image"], "hwf": features["hwf"]}


def ndc_rays(img_h, img_w, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
        img_h: height in pixels
        img_w: width in pixels
        focal: focal length of the pinhole camera
        near: near depth bound for the scene
        rays_o: (num_rays, 3) origin rays
        rays_d: (num_rays, 3) direction rays
    Returns:
        rays_o: (num_rays, 3) origin rays in NDC
        rays_d: (num_rays, 3) direction rays in NDC
    """
    # shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    ox = rays_o[..., 0] / rays_o[..., 2]
    oy = rays_o[..., 1] / rays_o[..., 2]

    # projection
    o0 = -1.0 / (float(img_w) / (2.0 * focal)) * ox
    o1 = -1.0 / (float(img_h) / (2.0 * focal)) * oy
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (float(img_w) / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox)
    d1 = -1.0 / (float(img_h) / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy)
    d2 = 1 - o2

    rays_o = tf.stack([o0, o1, o2], axis=-1)
    rays_d = tf.stack([d0, d1, d2], axis=-1)
    return rays_o, rays_d


def get_rays(img_h, img_w, focal, c2w):
    """Get ray origins and directions from a pinhole camera.
    Args:
        img_h: height in pixels
        img_w: width in pixels
        focal: focal length of the pinhole camera
        c2w: (3, 4) camera to world coordinate transformation matrix
    Returns:
        rays: (2, img_h * img_w, 3) stacked origin and direction rays
    """
    i, j = tf.meshgrid(tf.range(img_w), tf.range(img_h), indexing="xy")
    i = (tf.cast(i, dtype=tf.float32) - float(img_w) * 0.5) / float(focal)
    j = -(tf.cast(j, dtype=tf.float32) - float(img_h) * 0.5) / float(focal)
    dirs = tf.stack([i, j, -tf.ones_like(i, dtype=tf.float32)], axis=-1)
    rays_d = tf.einsum("ijl,kl", dirs, c2w[:3, :3])
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_dataset(data_dir, config, rng=None, cache=True, **kwargs):
    ds_dict = {
        "blender": load_blender.get_blender,
        "deepvoxels": load_deepvoxels.get_deepvoxels,
        "llff": load_llff.get_llff,
    }
    assert (
        config.dataset_type in ds_dict.keys()
    ), f"{config.dataset_type} is not one of 'blender', 'deepvoxels' or 'llff'."
    ds_output = ds_dict[config.dataset_type](data_dir, config, **kwargs)
    datasets, counts, optics, render_poses_ds, static_pose, num_poses = ds_output

    if rng is None:
        rngs = 2 * [[None, None]]
    else:
        rngs = list(jax.random.split(rng))

    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1

    datasets_list, counts_list = [], []
    rays_fn = functools.partial(prepare_rays, config=config)
    for ds, count, split in zip(datasets, counts, ["train", "val", "test"]):
        ds.with_options(options)
        if split == "train":
            ds = ds.map(rays_fn, num_parallel_calls=AUTOTUNE)  # load rays

            if config.batching:

                def map_reshape(features):
                    *_, chn = tf.unstack(tf.shape(features["rays"]))
                    rays = tf.reshape(features["rays"], [-1, chn])
                    image = tf.reshape(features["image"], [-1, 3])
                    return {"rays": rays, "image": image, "hwf": features["hwf"]}

                ds = ds.batch(count)
                ds = ds.map(map_reshape, num_parallel_calls=AUTOTUNE)
                ds = ds.cache().repeat()
                map_fn = functools.partial(batching_sample, config=config)
                if rng is None:
                    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
                else:
                    ds = deterministic_data._preprocess_with_per_example_rng(
                        ds, map_fn, rng=rngs.pop()
                    )
            else:
                if cache:
                    ds = ds.cache()
                ds = ds.repeat().shuffle(count, seed=rngs.pop()[0])
            ds = ds.batch(jax.local_device_count()).prefetch(AUTOTUNE)
            ds = prepare_train_data(ds)
        else:  # val or test
            ds = ds.repeat()
            if split == "val":
                ds = ds.shuffle(count, seed=rngs.pop()[0])
            ds = ds.map(rays_fn, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
            ds = iter(ds)
        datasets_list.append(ds)
        counts_list.append(count)

    ds_rays_fn = (
        lambda x: render_poses_ds.map(x, num_parallel_calls=AUTOTUNE)
        .repeat()
        .prefetch(AUTOTUNE)
    )
    render_rays_ds = iter(ds_rays_fn(rays_fn))

    render_rays_vdirs_ds = None
    if config.use_viewdirs:
        rays_fn = functools.partial(rays_fn, c2w_static_cam=static_pose)
        render_rays_vdirs_ds = iter(ds_rays_fn(rays_fn))
    render_datasets = render_rays_ds, render_rays_vdirs_ds, num_poses
    return datasets_list, counts_list, optics, render_datasets
