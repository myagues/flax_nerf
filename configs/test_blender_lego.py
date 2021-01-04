from configs import default as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    config.dataset_type = "blender"
    config.shape = "lego"
    config.down_factor = 2

    config.batching = False
    config.num_importance = 64
    config.num_rand = 1024
    config.num_samples = 64
    config.num_steps = 200000
    config.use_viewdirs = True
    config.white_bkgd = True

    config.i_print = 500
    config.i_img = 5000
    config.render_factor = 2

    return config
