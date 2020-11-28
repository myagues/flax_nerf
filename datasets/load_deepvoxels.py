import functools
import glob
import os

import imageio

import numpy as np
import pandas as pd

from absl import logging


def load_pose(file_path):
    assert os.path.isfile(file_path)
    return np.loadtxt(file_path, dtype=np.float32).reshape([4, 4])


def dir2poses(dataset_dir, testskip=None):
    posedir_file = sorted(glob.glob(os.path.join(dataset_dir, "pose", "*.txt")))
    poses = np.stack([load_pose(f) for f in posedir_file])
    transf = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    poses = poses @ transf
    poses = poses[:, :3, :4].astype(np.float32)
    if testskip is not None:
        return poses[::testskip]
    return poses


def get_imgs(dataset_dir, testskip=None):
    img_list = sorted(glob.glob(os.path.join(dataset_dir, "rgb", "*.png")))
    imgs = np.stack([imageio.imread(f) / 255.0 for f in img_list]).astype(np.float32)
    if testskip is not None:
        return imgs[::testskip]
    return imgs


def parse_intrinsics(dataset_dir, target_side_len, invert_y=False):
    # Get camera intrinsics
    filepath = os.path.join(dataset_dir, "intrinsics.txt")
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

    logging.info(
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


def load_dv_data(dataset_dir, scene="cube", testskip=8):
    dataset_subdir = lambda split: os.path.join(dataset_dir, split, scene)
    assert os.path.exists(dataset_subdir("train"))

    img_h = 512
    img_w = 512

    full_intrinsic, grid_barycenter, scale, near_plane = parse_intrinsics(
        dataset_subdir("train"), img_h
    )
    focal = full_intrinsic[0, 0]

    logging.info("Full intrinsic: %s", full_intrinsic)
    logging.info("Grid barycenter: %s", grid_barycenter)
    logging.info("Scale: %s", scale)
    logging.info("Near plane: %s", near_plane)
    # logging.info("World to camera poses: %s", w2c_poses)
    logging.info("Image height: %d, image width: %d, focal: %.5f", img_h, img_w, focal)

    train_poses = dir2poses(dataset_subdir("train"))
    val_poses = dir2poses(dataset_subdir("validation"), testskip=testskip)
    test_poses = dir2poses(dataset_subdir("test"), testskip=testskip)
    all_poses = np.concatenate([train_poses, val_poses, test_poses])
    del train_poses, val_poses

    train_imgs = get_imgs(dataset_subdir("train"))
    val_imgs = get_imgs(dataset_subdir("validation"), testskip=testskip)
    test_imgs = get_imgs(dataset_subdir("test"), testskip=testskip)

    all_imgs = np.concatenate([train_imgs, val_imgs, test_imgs])
    counts = [train_imgs.shape[0], val_imgs.shape[0], test_imgs.shape[0]]

    return all_imgs, all_poses, test_poses, (img_h, img_w, focal), counts
