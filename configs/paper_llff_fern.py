from configs import default as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    config.dataset_type = "llff"
    config.shape = "fern"
    config.llff.hold = 8
    config.down_factor = 4

    config.num_importance = 128
    config.num_rand = 4096
    config.num_samples = 64
    config.num_steps = 200000
    config.lr_decay = 250
    config.raw_noise_std = 1.0
    config.use_viewdirs = True
    config.num_poses = 120

    return config
