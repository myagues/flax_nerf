import ml_collections

from jax import numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    # fmt: off
    config = ml_collections.ConfigDict()

    config.learning_rate = 5e-4
    config.lr_schedule = "constant * decay_every"
    config.lr_decay = 250                # exponential learning rate decay (in 1000s)
    config.decay_factor = 0.1
    config.batching = True               # whether to only take random rays from all images or 1 image at a time
    config.num_steps = 1000000
    config.dtype = jnp.float32

    config.emb = ml_collections.ConfigDict()
    config.emb.use_embed = True          # whether to use default positional encoding
    config.emb.periodic_fns = (jnp.sin, jnp.cos)  # periodic functions to use for embedding
    config.emb.multires = 10             # log2 of max freq for positional encoding (3D location)
    config.emb.multires_views = 4        # log2 of max freq for positional encoding (2D direction)
    config.emb.log_sampling = True
    config.emb.include_input = True      # whether to include inputs with periodic functions

    config.model = ml_collections.ConfigDict()
    config.model.net_depth = 8           # layers in network
    config.model.net_width = 256         # channels per layer
    config.model.skips = (4,)

    config.model_fine = ml_collections.ConfigDict()
    config.model_fine.net_depth = 8      # layers in fine network
    config.model_fine.net_width = 256    # channels per layer in fine network
    config.model_fine.skips = (4,)

    # pre-crop options
    config.precrop_iters = 0             # number of steps to train on central crops
    config.precrop_frac = 0.5            # fraction of img taken for central crops

    # rendering options
    config.num_rand = 32 * 32 * 4        # batch size (number of random rays per gradient step)
    config.num_samples = 64              # number of coarse samples per ray
    config.num_importance = 0            # number of additional fine samples per ray
    config.perturb = True                # whether to use jitter
    config.use_viewdirs = True           # whether to use full 5D input instead of 3D
    config.raw_noise_std = 0.0           # std dev of noise added to regularize sigma_a output, 1e0 recommended
    config.render_only = False           # do not optimize, reload weights and render out render_poses path
    config.render_test = False           # render the test set instead of render_poses path
    config.down_factor = 1               # downsampling factor for the dataset
    config.render_factor = 1             # downsampling factor to speed up rendering, set 4 or 8 for fast preview
    config.white_bkgd = False            # whether to render synthetic data on a white bkgd (mandatory for dvoxels)

    # dataset options
    config.dataset_type = "llff"         # options: llff / blender / deepvoxels
    config.shape = "fern"                # scene in the dataset
    config.testskip = 8                  # will load 1/N images from test/val sets, useful for large datasets like deepvoxels
    config.num_poses = 40                # number of poses to generate for renders in "blender" and "llff"

    # llff flags
    config.llff = ml_collections.ConfigDict()
    config.llff.lindisp = False          # sampling linearly in disparity rather than depth
    config.llff.spherify = False         # set for spherical 360 scenes
    config.llff.hold = 8                 # will take every 1/N images as LLFF test set, paper uses 8

    # logging / saving options
    config.i_print = 100                 # frequency of console printout and metric logging
    config.i_img = 5000                  # frequency of TensorBoard image logging
    config.i_weights = 5000              # frequency of weight ckpt saving
    config.i_testset = 1000000           # frequency of testset saving
    config.i_video = 1000000             # frequency of render_poses video saving

    return config
