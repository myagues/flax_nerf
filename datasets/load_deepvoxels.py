import functools
import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from absl import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE

transf = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def parse_intrinsics(data_dir, target_side_len, invert_y=False):
    # Get camera intrinsics
    filepath = os.path.join(data_dir, "intrinsics.txt")
    df = pd.read_table(filepath, header=None, sep=" ", dtype=np.float32)
    f, cx, cy, _ = df.values[0]
    *grid_barycenter, _ = df.values[1]
    near_plane, *_ = df.values[2]
    scale, *_ = df.values[3]
    height, width, *_ = df.values[4]
    # try:
    #     world2cam_poses, *_ = df.values[5]
    # except ValueError:
    #     world2cam_poses = False
    logging.debug(
        "cx: %f, cy: %f, f: %f, height: %f, width: %f", cx, cy, f, height, width
    )

    cx /= width * target_side_len
    cy /= height * target_side_len
    fx = target_side_len / height * f
    fy = -f if invert_y else f

    # Build the intrinsic matrices
    full_intrinsic = np.array(
        [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ).astype(np.float32)
    return full_intrinsic, grid_barycenter, scale, near_plane


def get_near_far_vals(data_dir, config):
    def get_pose(pose_path):
        pose = tf.io.decode_csv(tf.io.read_file(pose_path), [0.0] * 16, field_delim=" ")
        pose = tf.reshape(pose, (4, 4)) @ transf
        pose = tf.cast(pose, tf.float32)
        return pose

    dataset_subdir = lambda split: os.path.join(data_dir, split, config.shape, "pose")
    item_size = 0
    for split in ["train", "validation", "test"]:
        posedir_file = sorted(glob.glob(os.path.join(dataset_subdir(split), "*.txt")))
        skip = 1 if split == "train" or config.testskip == 0 else config.testskip
        ds = tf.data.Dataset.from_tensor_slices(posedir_file[::skip])
        item_size += ds.cardinality().numpy()
        if split == "train":
            datasets = ds
        else:
            datasets = datasets.concatenate(ds)
    poses = datasets.map(get_pose, num_parallel_calls=AUTOTUNE).batch(item_size)

    hemi_R = np.mean(
        np.linalg.norm(np.concatenate(list(poses.as_numpy_iterator())), axis=-1)
    )
    near = hemi_R - 1.0
    far = hemi_R + 1.0
    return near, far


def get_image_data(file_path, pose_path, focal, factor):
    img = tf.io.decode_png(tf.io.read_file(file_path))
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    height, width, _ = tf.unstack(tf.shape(img))

    pose = tf.io.decode_csv(tf.io.read_file(pose_path), [0.0] * 16, field_delim=" ")
    pose = tf.reshape(pose, (4, 4)) @ transf
    pose = tf.cast(pose, tf.float32)

    if factor > 1:
        height //= factor
        width //= factor
        focal /= float(factor)
        img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.AREA)

    return {"image": img, "pose": pose, "hwf": (height, width, float(focal))}


def load_data(data_dir, split, config, img_h, img_w, factor):
    dataset_subdir = lambda split: os.path.join(data_dir, split, config.shape)
    assert os.path.exists(dataset_subdir(split))
    assert config.white_bkgd == True

    full_intrinsic, grid_barycenter, scale, near_plane = parse_intrinsics(
        dataset_subdir(split), img_h
    )
    focal = full_intrinsic[0, 0]

    logging.debug("Full intrinsic: %s", full_intrinsic)
    logging.debug("Grid barycenter: %s", grid_barycenter)
    logging.debug("Scale: %s", scale)
    logging.debug("Near plane: %s", near_plane)
    # logging.debug("World to camera poses: %s", w2c_poses)

    skip = 1 if split == "train" or config.testskip == 0 else config.testskip
    fnames = sorted(glob.glob(os.path.join(dataset_subdir(split), "rgb", "*.png")))
    poses = sorted(glob.glob(os.path.join(dataset_subdir(split), "pose", "*.txt")))

    map_fn = functools.partial(get_image_data, focal=focal, factor=factor)
    dataset = tf.data.Dataset.from_tensor_slices((fnames[::skip], poses[::skip]))
    counts = dataset.cardinality().numpy()
    dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
    return dataset, counts


def get_deepvoxels(data_dir, config):
    img_h = 512
    img_w = 512
    near, far = get_near_far_vals(data_dir, config)
    load_data_fn = functools.partial(
        load_data, data_dir, config=config, img_h=img_h, img_w=img_w
    )
    train_ds, train_items = load_data_fn("train", factor=config.down_factor)
    val_ds, val_items = load_data_fn("validation", factor=config.down_factor)
    test_ds, test_items = load_data_fn(
        "test", factor=(config.down_factor * config.render_factor)
    )
    render_example = next(iter(test_ds))
    static_pose = render_example["pose"].numpy()
    hwf = next(iter(train_ds))["hwf"]
    r_hwf = render_example["hwf"]

    datasets = train_ds, val_ds, test_ds
    counts = train_items, val_items, test_items
    optics = hwf, r_hwf, near, far
    return datasets, counts, optics, test_ds, static_pose, test_items
