from configs import default as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    config.dataset_type = "deepvoxels"
    config.shape = "cube"

    config.batching = False
    config.num_importance = 128
    config.num_rand = 4096
    config.num_samples = 64
    config.num_steps = 200000
    config.lr_decay = 200
    config.use_viewdirs = True
    config.white_bkgd = True

    return config
