import functools
import imageio
import json
import os

import jax

import numpy as np
import tensorflow as tf

from absl import logging

# make sure TF does not allocate gpu memory
tf.config.experimental.set_visible_devices([], "GPU")

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
    return c2w


def load_data(dataset_dir, half_res=True, testskip=1):
    splits = ["train", "val", "test"]

    all_imgs, all_poses = [], []
    counts = []
    for split in splits:
        with open(os.path.join(dataset_dir, f"transforms_{split}.json"), "r") as fp:
            meta = json.load(fp)

        imgs, poses = [], []
        skip = 1 if split == "train" or testskip == 0 else testskip
        logging.info(
            "Reading data for split %s: %d images", split, len(meta["frames"][::skip])
        )
        for frame in meta["frames"][::skip]:
            fname = os.path.join(dataset_dir, f"{frame['file_path']}.png")
            imgs.append(imageio.imread(fname))  # keep all 4 channels (RGBA)
            poses.append(np.array(frame["transform_matrix"]))

        all_imgs.append(imgs)
        all_poses.append(poses)
        counts.append(len(imgs))

    all_imgs = (np.concatenate(all_imgs) / 255.0).astype(np.float32)
    all_poses = np.concatenate(all_poses).astype(np.float32)

    img_height, img_width = all_imgs[0].shape[:2]
    focal = 0.5 * img_width / np.tan(0.5 * float(meta["camera_angle_x"]))

    if half_res:
        img_height //= 2
        img_width //= 2
        focal /= 2.0
        all_imgs = tf.image.resize(
            all_imgs, [img_height, img_width], method=tf.image.ResizeMethod.AREA
        )._numpy()

    render_poses = map(
        lambda angle: pose_spherical(angle, -30.0, 4.0),
        np.linspace(-180, 180, 40, endpoint=False),
    )
    render_poses = np.stack(list(render_poses))

    return all_imgs, all_poses, render_poses, (img_height, img_width, focal), counts
