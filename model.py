import functools
from typing import Any, Callable, Sequence

import numpy as np

from flax import linen as nn
from jax import numpy as jnp, lax, vmap


class NeRF(nn.Module):
    net_depth: int = 8
    net_width: int = 256
    skips: Sequence[int] = (4,)
    use_viewdirs: bool = True
    use_embed: bool = True
    multires: int = 10
    multires_views: int = 4
    periodic_fns: Sequence[Callable] = (jnp.sin, jnp.cos)
    log_sampling: bool = True
    include_input: bool = True
    input_channels: int = 3
    output_channels: int = 4
    dtype: Any = jnp.float32
    precision: Any = lax.Precision.DEFAULT

    def embed(self, inputs, multires):
        features_chn, _ = inputs.shape

        max_freq_log2 = multires - 1
        num_freqs = multires

        if self.log_sampling:
            freq_bands = 2.0 ** np.linspace(0.0, max_freq_log2, num_freqs)
        else:
            freq_bands = np.linspace(2.0 ** 0.0, 2.0 ** max_freq_log2, num_freqs)

        inputs_freq = vmap(lambda x: inputs * x)(freq_bands)
        fns = jnp.stack([fn(inputs_freq) for fn in self.periodic_fns])
        fns = fns.swapaxes(0, 2).reshape([features_chn, -1])

        if self.include_input:
            fns = jnp.concatenate([inputs, fns], axis=-1)
        return fns.astype(self.dtype)

    # @nn.remat
    @nn.compact
    def __call__(self, inputs_pts, inputs_views):
        assert inputs_pts.shape[2] == self.input_channels
        inputs_pts_shape = inputs_pts.shape
        inputs_pts = jnp.reshape(inputs_pts, [-1, self.input_channels])

        if self.use_embed:
            inputs_pts = self.embed(inputs_pts, self.multires)

        dense = functools.partial(nn.Dense, dtype=self.dtype, precision=self.precision)
        x = inputs_pts
        for i in range(self.net_depth):
            x = dense(self.net_width)(x)
            x = nn.relu(x)
            if i in self.skips:
                x = jnp.concatenate([x, inputs_pts], axis=-1)

        if self.use_viewdirs:
            assert inputs_views.shape[1] == self.input_channels
            inputs_views = jnp.broadcast_to(inputs_views[:, None], inputs_pts_shape)
            inputs_views = jnp.reshape(inputs_views, [-1, self.input_channels])

            if self.use_embed:
                inputs_views = self.embed(inputs_views, self.multires_views)

            alpha_out = dense(1)(x)
            bottleneck = dense(256)(x)
            inputs_viewdirs = jnp.concatenate([bottleneck, inputs_views], axis=-1)

            # "the supplement to the paper states there are 4 hidden layers here,
            # but this is an error since the experiments were actually run with
            # 1 hidden layer, so we will leave it as 1."
            x = inputs_viewdirs
            for i in range(1):
                x = dense(self.net_width // 2)(x)
                x = nn.relu(x)
            x = dense(3)(x)
            x = jnp.concatenate([x, alpha_out], axis=-1)
        else:
            x = dense(self.output_channels)(x)
        return x
