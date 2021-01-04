"""Some LLFF preprocessing functions taken from kwea123's NeRF implementation.
https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
"""

import functools
import glob
import os

import numpy as np
import tensorflow as tf

from absl import logging


AUTOTUNE = tf.data.experimental.AUTOTUNE
normalize = lambda x: x / np.linalg.norm(x)


def average_poses(poses):
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg


def recenter_poses(poses):
    """Center the poses so that we can use NDC."""
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    # convert to homogeneous coordinate for faster computation
    pose_avg_homo[:3] = pose_avg
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4)

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)
    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spheric_poses(radius, num_poses):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
        num_poses: int, number of poses to create along the path.
    Outputs:
        spheric_poses: (num_poses, 3, 4) the poses in the circular path.
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -0.9 * t],
                [0, 0, 1, t],
                [0, 0, 0, 1],
            ]
        )

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

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, num_poses, endpoint=False):
        spheric_poses += [
            spheric_pose(th, -np.pi / 5, radius)
        ]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def create_spiral_poses(radii, focus_depth, num_poses):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like: https://tinyurl.com/ybgtfns3
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        num_poses: int, number of poses to create along the path
    Outputs:
        poses_spiral: (num_poses, 3, 4) the poses in the spiral path
    """
    poses_spiral = []
    for t in np.linspace(
        0, 4 * np.pi, num_poses, endpoint=False
    ):  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)
    return np.stack(poses_spiral, 0)  # (num_poses, 3, 4)


def get_image_data(file_path, pose, factor, focal):
    img = tf.io.decode_png(tf.io.read_file(file_path))
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    height, width, _ = tf.unstack(tf.shape(img))

    if factor > 1:
        height //= factor
        width //= factor
        focal /= factor
        img = tf.image.resize(
            img,
            [height, width],
            method=tf.image.ResizeMethod.AREA,
            preserve_aspect_ratio=True,
        )
    return {"image": img, "pose": pose, "hwf": (height, width, float(focal))}


def load_data(data_dir, config, recenter=True, bd_factor=0.75, num_poses=120):

    fnames = sorted(glob.glob(os.path.join(data_dir, config.shape, "images", "*")))
    poses_arr = np.load(os.path.join(data_dir, config.shape, "poses_bounds.npy"))

    poses, bounds = np.split(poses_arr, [15], axis=-1)
    poses = poses.reshape([-1, 3, 5]).astype(np.float32)
    assert len(fnames) == poses.shape[0]
    *_, focal = poses[0, :, -1]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

    if recenter:
        poses, _ = recenter_poses(poses)
    distances_from_center = np.linalg.norm(poses[..., 3], axis=1)
    val_idx = np.argmin(distances_from_center)

    # Rescale if bd_factor is provided
    scale_factor = 1.0 if bd_factor is None else (bounds.min() * bd_factor)
    bounds /= scale_factor
    poses[..., 3] /= scale_factor

    val_fnames, val_poses = fnames[val_idx], poses[val_idx, None].astype(np.float32)
    fnames.pop(val_idx)
    poses = np.delete(poses, val_idx, 0)
    logging.info("Validation image is: %s", val_fnames)

    split_idx = np.arange(len(poses))[:: config.llff.hold]
    test_fnames = fnames[:: config.llff.hold]
    test_poses = np.take(poses, split_idx, axis=0).astype(np.float32)

    train_fnames = np.delete(fnames, split_idx, 0)
    train_poses = np.delete(poses, split_idx, 0).astype(np.float32)

    if config.llff.spherify:
        near = bounds.min()
        far = min(8 * near, bounds.max())  # focus on central object only
        radius = 1.1 * bounds.min()
        render_poses = create_spheric_poses(radius, num_poses)
    else:
        near, far = 0, 1
        # hardcoded, this is numerically close to the formula given in the original repo
        # mathematically if near=1 and far=infinity, then this number will converge to 4
        focus_depth = 3.5
        radii = np.percentile(np.abs(poses[..., 3]), 90, axis=0)
        render_poses = create_spiral_poses(radii, focus_depth, num_poses)

    map_fn = functools.partial(
        get_image_data,
        factor=config.down_factor,
        focal=focal,
    )
    train_ds = tf.data.Dataset.from_tensor_slices((train_fnames, train_poses))
    train_ds = train_ds.map(map_fn, num_parallel_calls=AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(([val_fnames], val_poses))
    val_ds = val_ds.map(map_fn, num_parallel_calls=AUTOTUNE)

    map_fn = functools.partial(
        get_image_data,
        factor=(config.down_factor * config.render_factor),
        focal=focal,
    )
    test_ds = tf.data.Dataset.from_tensor_slices((test_fnames, test_poses))
    test_ds = test_ds.map(map_fn, num_parallel_calls=AUTOTUNE)

    datasets = train_ds, val_ds, test_ds
    counts = map(lambda x: x.cardinality().numpy(), datasets)
    return datasets, counts, render_poses.astype(np.float32), near, far


def get_llff(data_dir, config, **kwargs):
    datasets, counts, render_poses, near, far = load_data(data_dir, config, **kwargs)
    train_ds, _, test_ds = datasets
    hwf = next(iter(train_ds))["hwf"]
    r_hwf = next(iter(test_ds))["hwf"]
    optics = hwf, r_hwf, near, far

    render_poses_ds = tf.data.Dataset.from_tensor_slices(list(render_poses))
    static_pose = next(iter(render_poses_ds))
    render_poses_ds = render_poses_ds.map(
        lambda x: {"image": False, "pose": x, "hwf": r_hwf}, num_parallel_calls=AUTOTUNE
    )
    num_poses = len(render_poses)
    return datasets, counts, optics, render_poses_ds, static_pose.numpy(), num_poses
