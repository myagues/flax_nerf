from configs import default as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    config.dataset_type = "blender"
    config.shape = "lego"

    config.batching = False
    config.num_importance = 128
    config.num_rand = 1024
    config.num_samples = 64
    config.num_steps = 500000
    config.lr_decay = 500
    config.use_viewdirs = True
    config.white_bkgd = True

    return config
