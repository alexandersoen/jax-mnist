from functools import partial

import flax.nnx as nnx


class CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=32,
            kernel_size=(3, 3),
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(3, 3),
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=3136,
            out_features=256,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=256,
            out_features=10,
            rngs=rngs,
        )

        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x
