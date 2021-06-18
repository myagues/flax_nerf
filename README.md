# Neural Radiance Fields (NeRF) with Flax

This repository is an unofficial implementation of *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, using [Flax](https://github.com/google/flax) and the [Linen API](https://github.com/google/flax/tree/master/flax/linen).

B. Mildenhall, P.P. Srinivasan, M. Tancik, J.T. Barron, R. Ramamoorthi and R. Ng, *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, 2020, ECCV, [arXiv:2003.08934 [cs.CV]](https://arxiv.org/abs/2003.08934).

Original repository can be found in [bmild/nerf](https://github.com/bmild/nerf).

## Description

Neural Radiance Fields (NeRF) is a method for synthesizing novel views of complex scenes, by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Views are synthesized by querying 5D coordinates (spatial location (*x*, *y*, *z*) and viewing direction (*θ*, *ϕ*)) along camera rays and using classic volume rendering techniques to project the output colors and densities into an image.

This implementation tries to be as close as possible to the original source, bringing some code optimizations and using the flexibility and native multi device (GPUs and TPUs) support in JAX.

Most of the comments are from the original work, which are very helpful for understanding the model steps.

## Installation

Install `jax` and `jaxlib` according to your [platform configuration](https://github.com/google/jax#installation). Then, install the necessary dependencies with:

```
pip install --upgrade clu flax imageio imageio-ffmpeg ml_collections optax pandas tensorboard 'tensorflow>=2.4' tqdm
```

## Data

There are three subsets of data used in the original publication that can be downloaded from [nerf_data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1):
- [Blender](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) from NeRF authors (`nerf_synthetic.zip`)
- [DeepVoxels](https://drive.google.com/open?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl) from Vincent Sitzmann (`nerf_real_360.zip`)
- [LLFF](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) from NeRF authors (`nerf_llff_data.zip`)

In addition, there is:
- [nerf_example_data](https://drive.google.com/open?id=1xzockqgkO-H3RCGfkZvIZNjOnk3l7AcT) is limited to the `lego` (from Blender) and `fern` (from LLFF) scenes
- [tiny_nerf_data](https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz) is a low resolution `lego` used in the simplified notebook example

## How to run

Required parameters to run the training are:
- `--data_dir`: directory where data is placed
- `--model_dir`: model saving location
- `--config`: configuration parameters

```
python main.py \
    --data_dir=/data/nerf_synthetic \
    --model_dir=logs \
    --config=configs/test_blender_lego.py
```

Configuration flag is defined using [`config_flags`](https://github.com/google/ml_collections/tree/master#config-flags), which allows overriding configuration fields, and can be done as follows:

```
python main.py \
    --data_dir=/data/nerf_synthetic \
    --model_dir=logs \
    --config=configs/test_blender_lego.py \
    --config.num_samples=128 \
    --config.i_print=250
```

__NOTE__: check and understand the effect of default parameters in `configs/default.py` to avoid confusion when passing arguments to the model.

## Examples

All examples were run on an NVIDIA RTX 2080Ti. Examples prior to deterministic datasets are available in [e81d608](https://github.com/myagues/flax_nerf/tree/e81d608128d92c8d15fd17c21834a5e47c185359).

### Blender - `lego`

<details>
    <summary>Commands</summary>

```
python main.py \
    --data_dir=/data/nerf_synthetic \
    --model_dir=logs_lego_64 \
    --config=configs/test_blender_lego.py \
    --config.batching=True \
    --config.i_img=10000 \
    --config.i_weights=10000

python render.py \
    --data_dir=/data/nerf_synthetic \
    --model_dir=logs_lego_64 \
    --config=configs/test_blender_lego.py \
    --config.render_factor=1 \
    --config.testskip=0 \
    --render_video_set=test
```

```
python main.py \
    --data_dir=/data/nerf_synthetic \
    --model_dir=logs_lego_128 \
    --config=configs/paper_blender_lego.py \
    --config.batching=True \
    --config.i_img=10000 \
    --config.i_weights=10000

python render.py \
    --data_dir=/data/nerf_synthetic \
    --model_dir=logs_lego_128 \
    --config=configs/paper_blender_lego.py \
    --config.render_factor=1 \
    --config.testskip=0 \
    --render_video_set=test
```
</details>

Checkpoint path | Test set PSNR | Paper PSNR | TensorBoard.dev
:---------------: | :-------------: |  :-------------: | ---------------:
[lego_400_64](https://drive.google.com/drive/folders/1d4bseKj_lBzhszqrY46402xUPxJ0rNyB?usp=sharing) | 31.48 | - | [2021-01-15](https://tensorboard.dev/experiment/kw5Sqp64S5akhnpku3LgAQ)
[lego_800_128](https://drive.google.com/drive/folders/1AJ3h2k9cXUZdKz7In1U7MW8cDVzU-dbM?usp=sharing) | 32.29 | 32.54 | [2021-01-15](https://tensorboard.dev/experiment/kw5Sqp64S5akhnpku3LgAQ)

## Tips and caveats

- You can test or debug multiple devices in a **CPU only** installation using `XLA_FLAGS` environment variable (more information in [JAX #1408](https://github.com/google/jax/issues/1408)). To simulate 4 devices:

```
XLA_FLAGS="--xla_force_host_platform_device_count=4 xla_cpu_multi_thread_eigen=False intra_op_parallelism_threads=1"
```

- Try to minimize time spent on rendering intermediate results during training (`i_video`, `i_testset`) and rely on validation results in TensorBoard. Either save intermediate checkpoints and render after training or use `render_factor` and `testskip` to your advantage.

- Here are some recommendations for reducing GPU memory footprint:
    - Use `nn.remat` decorator in your network module (more about `jax.remat` in [JAX #1749](https://github.com/google/jax/pull/1749))
    - Decrease model parameters (`net_depth`, `net_width`, `num_importance`, `num_rand`, `num_samples`)
    - Using `bfloat16` will decrease memory usage by half, but the low precision reduces performance by a big margin

- The original repository ([bmild/nerf/issues](https://github.com/bmild/nerf/issues)) has many good comments and explanations from the authors and participants, which help to better understand the limitations and applications for this approach

- [kwea123/nerf_pl](https://github.com/kwea123/nerf_pl) is another implementation, using PyTorch Lightning, that has many explanations and applications for your trained models

- [google/jaxnerf](https://github.com/google-research/google-research/tree/master/jaxnerf) is _kind of_ an official version of NeRF with JAX and Flax

- Training these models in Colab with TPUs is a bit of a stretch ([FAQ - Resource Limits](https://research.google.com/colaboratory/faq.html#resource-limits)), although you can use it for rendering (800px square image takes ~26s in an NVIDIA RTX 2080Ti vs ~7s in a TPUv2). Add the following commands to the top of your file:

```
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
```

## TODO

- Rendering routines use `lax.map`, which is convenient for shaping outputs and fast at execution, although reshaping is a nuisance in some cases. Wait for mask redesign or rethink the execution.
- Most of the processes are done with batches of rays, rewrite everything for a single ray and `vmap/pmap/xmap` as needed (wait for JAX unified map API [JAX#2939](https://github.com/google/jax/issues/2939)).
- Add function docs and lint
