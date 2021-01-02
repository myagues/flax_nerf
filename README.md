# Neural Radiance Fields (NeRF) with Flax

This repository is an unofficial implementation of *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, using [Flax](https://github.com/google/flax) and the [Linen API](https://github.com/google/flax/tree/master/flax/linen).

B. Mildenhall, P.P. Srinivasan, M. Tancik, J.T. Barron, R. Ramamoorthi and R. Ng, *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, 2020, ECCV, [arXiv:2003.08934 [cs.CV]](https://arxiv.org/abs/2003.08934).

Original repository can be found in [bmild/nerf](https://github.com/bmild/nerf).

## Description

Neural Radiance Fields (NeRF) is a method for synthesizing novel views of complex scenes, by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Views are synthesized by querying 5D coordinates (spatial location (*x*, *y*, *z*) and viewing direction (*θ*, *ϕ*)) along camera rays and using classic volume rendering techniques to project the output colors and densities into an image.

This implementation tries to be as close as possible to the original source, bringing some code optimizations and using the flexibility and native multi device (GPUs and TPUs) support JAX offers.

Most of the comments are from the original work, which are very helpful for understanding the model steps.

## Installation

Install `jax` and `jaxlib` according to your [platform configuration](https://github.com/google/jax#installation). Then, install the necessary dependencies with:

```
pip install clu imageio imageio-ffmpeg flax ml_collections pandas tensorboard tensorflow
```

## Data

There are three subsets of data used in the original publication that can be downloaded from [nerf_data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1):
- [Blender](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) aka nerf_synthetic (NeRF authors)
- [DeepVoxels](https://drive.google.com/open?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl) (Vincent Sitzmann)
- [LLFF](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) (NeRF authors)

In addition, there is:
- [nerf_example_data](https://people.eecs.berkeley.edu/~bmild/nerf/nerf_example_data.zip) is limited to the `lego` (from Blender) and `fern` (from LLFF) scenes
- [tiny_nerf_data](https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz) is a low resolution `lego` used in the simplified notebook example

## How to run

Required parameters to run the training are:
- `--data_dir`: directory where data is place
- `--model_dir`: model saving location
- `--config`: configuration parameters

```
python main.py \
    --data_dir=/data/nerf_synthetic/lego \
    --model_dir=logs \
    --config=configs/test_blender_lego.py
```

Configuration flag is defined using [`config_flags`](https://github.com/google/ml_collections/tree/master#config-flags), which allows overriding configuration fields, and can be done as follows:

```
python main.py \
    --data_dir=/data/nerf_synthetic/lego \
    --model_dir=logs \
    --config=configs/test_blender_lego.py \
    --config.num_samples=128 \
    --config.i_print=250
```

## Examples

All these examples were run on an NVIDIA RTX 2080 Ti with commit [e81d608](https://github.com/myagues/flax_nerf/tree/e81d608128d92c8d15fd17c21834a5e47c185359):

- `lego` scene from the Blender dataset with `--config.num_importance={0,64}`:

```
python main.py \
    --data_dir=/data/nerf_synthetic/lego \
    --model_dir=logs_lego_64 \
    --config=configs/test_blender_lego.py

python render.py \
    --data_dir=/data/nerf_synthetic/lego \
    --model_dir=logs_lego_64 \
    --config=configs/test_blender_lego.py \
    --config.render_factor=0 \
    --config.testskip=0 \
    --render_video_set=test
```

```
python main.py \
    --data_dir=/data/nerf_synthetic_lego \
    --model_dir=logs_lego_0 \
    --config=configs/test_blender_lego.py \
    --config.num_importance=0

python render.py \
    --data_dir=/data/nerf_synthetic/lego \
    --model_dir=logs_lego_0 \
    --config=configs/test_blender_lego.py \
    --config.num_importance=0 \
    --config.render_factor=0 \
    --config.testskip=0 \
    --render_video_set=test
```

Checkpoint path | Test set PSNR | Test set loss | TensorBoard.dev
:---------------: | :-------------: | :-------------: |---------------:
[lego_ckpt_0](https://drive.google.com/drive/folders/1h0r4ePMLueGExAqWJvWKUJORf9ju3XCF?usp=sharing) | 26.544 | 2.2e-3 | [2020-11-30](https://tensorboard.dev/experiment/WsKI4cYQS8OKDCMPFGPOLA)
[lego_ckpt_64](https://drive.google.com/drive/folders/1gM3eVfYQgYLsCqUDHxv0I0N3BfgAHlWE?usp=sharing) | 31.779 | 6.6e-4 | [2020-11-30](https://tensorboard.dev/experiment/WsKI4cYQS8OKDCMPFGPOLA)

- `vase` scene from the DeepVoxels dataset:

```
python main.py \
    --data_dir=/data/deepvoxels \
    --model_dir=logs_dv_vase \
    --config=configs/test_dvox_greek.py \
    --config.shape=vase \
    --config.num_importance=128 \
    --config.precrop_iters=500 \
    --config.num_steps = 200000
    --config.i_video=300000 \
    --config.i_testset=300000 \
    --config.i_img=5000 \
    --config.i_print=250

python render.py \
    --data_dir=/data/deepvoxels \
    --model_dir=logs_dv_vase \
    --config=configs/test_dvox_greek.py \
    --config.shape=vase \
    --config.num_importance=128 \
    --config.render_factor=0 \
    --config.testskip=4 \
    --render_video=False
```

Checkpoint path | Test set PSNR* | Test set loss* | TensorBoard.dev
:---------------: | :-------------: | :-------------: |---------------:
[vase_ckpt](https://drive.google.com/drive/folders/1yFquenIxwG2BfMrHOgiYZ_T9bSG8iUIN?usp=sharing) | 35.328 | 2.9e-4 | [2020-11-30](https://tensorboard.dev/experiment/HJN9sZNPQ9mknFNuVHkKkA)

*Only a subset of the test set has been used, given that DeepVoxels has a big amount of images

## Tips and caveats

- You can test or debug multiple devices in a **CPU only** machine using `XLA_FLAGS` environment variable (more information in [JAX #1408](https://github.com/google/jax/issues/1408)). To simulate 4 devices:

```
XLA_FLAGS="--xla_force_host_platform_device_count=4 xla_cpu_multi_thread_eigen=False intra_op_parallelism_threads=1"
```

- Rendering images is done using `lax.map`, which means that the image size must be divisible by the number of devices

- Try to minimize time spent on rendering intermediate results during training (`i_video`, `i_testset`) and rely on validation results in TensorBoard. Either save intermediate checkpoints and render after training or use `render_factor` and `testskip`

- This implementation does not chunk the batch at training, which makes it less flexible for GPUs with small memory capacity. Here are some recommendations:

    - Use `nn.remat` decorator in your network module (more about `jax.remat` in [JAX #1749](https://github.com/google/jax/pull/1749))
    - Decrease model parameters (`net_depth`, `net_width`, `num_importance`, `num_rand`, `num_samples`)
    - Use `--config.batching=True` to load a single training image per step, instead of precomputing all training rays upfront
    - Using `bfloat16` will decrease memory usage by half, but reduces the performance results by a big margin, so it is **not** an option to use (no tests have been made with `float16`)

- The original repository ([bmild/nerf/issues](https://github.com/bmild/nerf/issues)) has many good comments and explanations from the authors and participants, which help to better understand the limitations and applications for this approach

- [kwea123/nerf_pl](https://github.com/kwea123/nerf_pl) is another implementation, using PyTorch Lightning, that has many explanations and applications for your trained models

## TODO

- Add LLFF data reader
- Rendering routines use `lax.map`, which is problematic if image size is not divisible by the number of devices. Try using mask and padding for a more flexible implementation.
- Redo DeepVoxels data reader with `tf.data.Dataset`
- Most of the processes are done with batches of rays, rewrite everything for a single ray and `vmap/pmap/xmap` as needed (wait for JAX unified map API [JAX#2939](https://github.com/google/jax/issues/2939))
