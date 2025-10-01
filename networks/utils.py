from functools import partial
from typing import Any, Callable

import flax.linen as nn
from flax.linen.initializers import constant, variance_scaling
from jax import numpy as jnp
from jax._src import dtypes

Initializer = Callable[..., Any]
Array = Any


activations = {
    "Linear": lambda x: x,
    "ReLU": nn.relu,
    "ELU": nn.elu,
    "Softplus": nn.softplus,
    "LeakyReLU": nn.leaky_relu,
    "Tanh": jnp.tanh,
    "Sigmoid": nn.sigmoid,
    "Exp": jnp.exp,
}


def torch_kernel_init(scale=1.0 / 3.0):
    return partial(
        variance_scaling, scale=scale, mode="fan_in", distribution="uniform"
    )()


def linspace_bias_init(scale):
    def init(key, shape, dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return jnp.linspace(
            start=-jnp.sqrt(3) * scale,
            stop=jnp.sqrt(3) * scale,
            num=shape[0],
            dtype=dtype,
        )

    return init


class Elephant(nn.Module):
    """Elephant activation function."""

    sigma: float = 1.0
    d: int = 2
    h: float = 1.0

    @nn.compact
    def __call__(self, x) -> Array:
        log_sigma = self.param("log_sigma", constant(jnp.log(self.sigma)), x.shape[1:])
        log_h = self.param("log_h", constant(jnp.log(self.h)), x.shape[1:])
        y = jnp.exp(log_h) / (1.0 + jnp.power(jnp.abs(x / jnp.exp(log_sigma)), self.d))
        return y


class Maxout(nn.Module):
    """Maxout activation function."""

    k: int = 2

    @nn.compact
    def __call__(self, x) -> Array:
        y = x.reshape(x.shape[:-1] + (-1, self.k))
        y = y.max(axis=-1)
        return y


class LWTA(nn.Module):
    """Winner take all activation function."""

    k: int = 2

    @nn.compact
    def __call__(self, x) -> Array:
        y = x.reshape(x.shape[:-1] + (-1, self.k))
        idx = y.argmax(axis=-1)
        indices = tuple(jnp.indices(y.shape[:-1])) + (idx,)
        m = jnp.zeros_like(y)
        m = m.at[indices].set(1)
        y = jnp.multiply(m, y).reshape(x.shape)
        return y


class FTA(nn.Module):
    """Fuzzy tiling activation function.
    Reference: https://github.com/hwang-ua/fta_pytorch_implementation
    """

    k: int = 20  # Number of tiles
    bound: float = 20.0

    def setup(self):
        assert self.k > 0, "The number of tiles must be positive."
        self.eta = 2 * self.bound / self.k
        self.delta = self.eta
        self.c_mat = self.delta * jnp.array(range(self.k)) - self.bound

    def relu_sum(self, c, x) -> Array:
        out = nn.relu(c - x) + nn.relu(x - self.delta - c)
        return out

    def I_eta_plus(self, x) -> Array:
        out = (x <= self.eta) * x + (x > self.eta) * 1.0
        return out

    def __call__(self, z) -> Array:
        z = jnp.expand_dims(z, axis=-1)
        out = 1.0 - self.I_eta_plus(self.relu_sum(self.c_mat, z))
        out = out.reshape(z.shape[:-2] + (-1,))
        return out
