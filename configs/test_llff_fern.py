from configs import default as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    config.dataset_type = "llff"

    config.llff.factor = 8
    config.llff.hold = 8

    config.num_importance = 64
    config.num_rand = 1024
    config.num_samples = 64
    config.raw_noise_std = 1.0
    config.use_viewdirs = True

    return config
