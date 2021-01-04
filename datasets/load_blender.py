import functools
import json
import os

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

trans_t = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]])

rot_phi = lambda phi: np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
)

rot_theta = lambda th: np.array(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w.astype(np.float32)


def get_image_data(file_path, pose, factor, camera_angle_x, white_bkgd):
    img = tf.io.decode_png(tf.io.read_file(file_path))
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    height, width, _ = tf.unstack(tf.shape(img))
    focal = 0.5 * float(height) / tf.math.tan(0.5 * camera_angle_x)

    if factor > 1:
        height //= factor
        width //= factor
        focal /= float(factor)
        img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.AREA)

    if white_bkgd:
        img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
    else:
        img = img[..., :3]
    return {"image": img, "pose": pose, "hwf": (height, width, focal)}


def load_data(data_dir, split, config, factor):

    with open(os.path.join(data_dir, f"transforms_{split}.json"), "r") as fp:
        meta = json.load(fp)

    skip = 1 if split == "train" or config.testskip == 0 else config.testskip

    camera_angle_x = float(meta["camera_angle_x"])
    fnames = []
    poses = []
    for frame in meta["frames"][::skip]:
        fnames.append(os.path.join(data_dir, f"{frame['file_path']}.png"))
        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

    map_fn = functools.partial(
        get_image_data,
        factor=factor,
        camera_angle_x=camera_angle_x,
        white_bkgd=config.white_bkgd,
    )
    dataset = tf.data.Dataset.from_tensor_slices((fnames, poses))
    counts = dataset.cardinality().numpy()
    dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
    return dataset, counts


def get_blender(data_dir, config, num_poses=40):
    near = 2.0
    far = 6.0
    data_dir = os.path.join(data_dir, config.shape)
    load_data_fn = functools.partial(load_data, data_dir, config=config)
    train_ds, train_items = load_data_fn("train", factor=config.down_factor)
    val_ds, val_items = load_data_fn("val", factor=config.down_factor)
    test_ds, test_items = load_data_fn(
        "test", factor=(config.down_factor * config.render_factor)
    )

    datasets = [train_ds, val_ds, test_ds]
    counts = [train_items, val_items, test_items]
    hwf = next(iter(train_ds))["hwf"]
    r_hwf = next(iter(test_ds))["hwf"]
    optics = hwf, r_hwf, near, far

    angles = np.linspace(-180, 180, num_poses, endpoint=False)
    render_poses = map(lambda x: pose_spherical(x, -30.0, 4.0), angles)
    render_poses_ds = tf.data.Dataset.from_tensor_slices(list(render_poses))
    static_pose = next(iter(render_poses_ds))
    render_poses_ds = render_poses_ds.map(
        lambda x: {"image": False, "pose": x, "hwf": r_hwf}, num_parallel_calls=AUTOTUNE
    )
    return datasets, counts, optics, render_poses_ds, static_pose.numpy(), num_poses
