from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from networks.utils import (
    FTA,
    LWTA,
    Elephant,
    Initializer,
    Maxout,
    activations,
    linspace_bias_init,
    torch_kernel_init,
)


class RLMLP(nn.Module):
    """(Elephant) multilayer perceptron for RL tasks."""

    hidden_dims: Sequence[int] = (1000,)
    output_dim: int = 10
    hidden_act: str = "ReLU"
    kernel_init: Initializer = torch_kernel_init
    bias_init: Initializer = linspace_bias_init
    last_w_scale: float = 1.0 / 3.0
    # Elephant
    d: int = 4
    sigma: float = 0.1
    bias_std: float = 0.0
    h: float = 1.0
    # Maxout
    maxout_k: int = -1
    # LWTA
    lwta_k: int = -1
    # FTA
    fta_k: int = -1
    fta_bound: float = -1.0
    # Last layer
    keep_last_layer: bool = True

    def setup(self):
        # Set feature network
        layers = []
        hidden_dims = list(self.hidden_dims)
        if self.hidden_act == "Elephant":
            for i in range(len(hidden_dims)):
                layers.append(
                    nn.Dense(
                        hidden_dims[i],
                        kernel_init=self.kernel_init(),
                        bias_init=self.bias_init(self.bias_std),
                        use_bias=True,
                    )
                )
                # Activation normalization
                layers.append(nn.LayerNorm())
                # Elephant activation
                layers.append(Elephant(sigma=self.sigma, d=self.d, h=self.h))
        elif self.hidden_act == "Maxout":
            for i in range(len(hidden_dims)):
                layers.append(
                    nn.Dense(
                        hidden_dims[i], kernel_init=self.kernel_init(), use_bias=True
                    )
                )
                layers.append(Maxout(k=self.maxout_k))
        elif self.hidden_act == "LWTA":
            for i in range(len(hidden_dims)):
                layers.append(
                    nn.Dense(
                        hidden_dims[i], kernel_init=self.kernel_init(), use_bias=True
                    )
                )
                layers.append(LWTA(k=self.lwta_k))
        elif self.hidden_act == "FTA":
            for i in range(len(hidden_dims)):
                layers.append(
                    nn.Dense(
                        hidden_dims[i], kernel_init=self.kernel_init(), use_bias=True
                    )
                )
                layers.append(activations["ReLU"])
            layers.pop(-1)
            layers.append(FTA(k=self.fta_k, bound=self.fta_bound))
        else:
            for i in range(len(hidden_dims)):
                layers.append(
                    nn.Dense(
                        hidden_dims[i], kernel_init=self.kernel_init(), use_bias=True
                    )
                )
                layers.append(activations[self.hidden_act])
        if self.keep_last_layer:
            layers.append(
                nn.Dense(
                    self.output_dim,
                    kernel_init=self.kernel_init(self.last_w_scale),
                    use_bias=True,
                )
            )
        self.mlp = nn.Sequential(layers)

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.mlp(x)
        return x


class DQNNet(nn.Module):
    """DQN (elephant) network."""

    hidden_dims: Sequence[int] = (64, 64)
    action_size: int = 10
    hidden_act: str = "ReLU"
    kernel_init: Initializer = torch_kernel_init
    bias_init: Initializer = linspace_bias_init
    # Elephant
    d: int = 4
    sigma: float = 0.1
    bias_std: float = 0.0
    h: float = 1.0
    # Maxout
    maxout_k: int = -1
    # LWTA
    lwta_k: int = -1
    # FTA
    fta_k: int = -1
    fta_bound: float = -1.0

    def setup(self):
        self.Q_net = RLMLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.action_size,
            hidden_act=self.hidden_act,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            d=self.d,
            sigma=self.sigma,
            bias_std=self.bias_std,
            h=self.h,
            maxout_k=self.maxout_k,
            lwta_k=self.lwta_k,
            fta_k=self.fta_k,
            fta_bound=self.fta_bound,
        )

    def __call__(self, obs):
        q = self.Q_net(obs)
        return q


class MLPGaussianTanhActor(nn.Module):
    """MLP actor network with Guassian policy N(mu, std): Tanh is applied outside of this module."""

    action_size: int = 4
    hidden_dims: Sequence[int] = (32, 32)
    hidden_act: str = "ReLU"
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    kernel_init: Initializer = torch_kernel_init
    bias_init: Initializer = linspace_bias_init
    # Elephant
    d: int = 4
    sigma: float = 0.1
    bias_std: float = 0.0
    h: float = 1.0
    # Maxout
    maxout_k: int = -1
    # LWTA
    lwta_k: int = -1
    # FTA
    fta_k: int = -1
    fta_bound: float = -1.0

    def setup(self):
        self.actor_net = RLMLP(
            hidden_dims=self.hidden_dims,
            output_dim=2 * self.action_size,
            hidden_act=self.hidden_act,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            d=self.d,
            sigma=self.sigma,
            bias_std=self.bias_std,
            h=self.h,
            maxout_k=self.maxout_k,
            lwta_k=self.lwta_k,
            fta_k=self.fta_k,
            fta_bound=self.fta_bound,
        )

    def __call__(self, obs):
        u_mean, u_log_std = jnp.split(
            self.actor_net(obs), indices_or_sections=2, axis=-1
        )
        u_log_std = jnp.clip(u_log_std, self.log_std_min, self.log_std_max)
        return u_mean, u_log_std


class MLPQCritic(nn.Module):
    """MLP action value critic: Q(s,a)."""

    hidden_dims: Sequence[int] = (32, 32)
    hidden_act: str = "ReLU"
    kernel_init: Initializer = torch_kernel_init
    bias_init: Initializer = linspace_bias_init
    # Elephant
    d: int = 4
    sigma: float = 0.1
    bias_std: float = 0.0
    h: float = 1.0
    # Maxout
    maxout_k: int = -1
    # LWTA
    lwta_k: int = -1
    # FTA
    fta_k: int = -1
    fta_bound: float = -1.0

    def setup(self):
        self.Q_net = RLMLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            hidden_act=self.hidden_act,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            d=self.d,
            sigma=self.sigma,
            bias_std=self.bias_std,
            h=self.h,
            maxout_k=self.maxout_k,
            lwta_k=self.lwta_k,
            fta_k=self.fta_k,
            fta_bound=self.fta_bound,
        )

    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], -1)
        q = self.Q_net(x)
        return q


class Temperature(nn.Module):
    """Self-tuning temperature for SAC."""

    init_temp: float = 1.0

    def setup(self):
        self.log_temp = self.param(
            "log_temp", init_fn=lambda seed: jnp.full((), jnp.log(self.init_temp))
        )

    def __call__(self):
        return jnp.exp(self.log_temp)


class MLPGaussianActor(nn.Module):
    """MLP actor network with Guassian policy N(mu, std)."""

    action_size: int = 4
    hidden_dims: Sequence[int] = (32, 32)
    hidden_act: str = "ReLU"
    kernel_init: Initializer = torch_kernel_init
    bias_init: Initializer = linspace_bias_init
    # Elephant
    d: int = 4
    sigma: float = 0.1
    bias_std: float = 0.0
    h: float = 1.0
    # Maxout
    maxout_k: int = -1
    # LWTA
    lwta_k: int = -1
    # FTA
    fta_k: int = -1
    fta_bound: float = -1.0

    def setup(self):
        self.actor_feature = RLMLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.action_size,
            hidden_act=self.hidden_act,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            d=self.d,
            sigma=self.sigma,
            bias_std=self.bias_std,
            h=self.h,
            maxout_k=self.maxout_k,
            lwta_k=self.lwta_k,
            fta_k=self.fta_k,
            fta_bound=self.fta_bound,
            keep_last_layer=False,
        )
        self.actor_mean = nn.Dense(self.action_size, kernel_init=self.kernel_init())
        self.actor_std = nn.Sequential(
            [nn.Dense(self.action_size, kernel_init=self.kernel_init()), nn.sigmoid]
        )

    def __call__(self, obs):
        feature = self.actor_feature(obs)
        u_mean = self.actor_mean(feature)
        u_std = self.actor_std(feature)
        return u_mean, u_std


class MLPVCritic(nn.Module):
    """MLP state value critic: V(s)."""

    hidden_dims: Sequence[int] = (32, 32)
    hidden_act: str = "ReLU"
    kernel_init: Initializer = torch_kernel_init
    bias_init: Initializer = linspace_bias_init
    # Elephant
    d: int = 4
    sigma: float = 0.1
    bias_std: float = 0.0
    h: float = 1.0
    # Maxout
    maxout_k: int = -1
    # LWTA
    lwta_k: int = -1
    # FTA
    fta_k: int = -1
    fta_bound: float = -1.0

    def setup(self):
        self.V_net = RLMLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            hidden_act=self.hidden_act,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            d=self.d,
            sigma=self.sigma,
            bias_std=self.bias_std,
            h=self.h,
            maxout_k=self.maxout_k,
            lwta_k=self.lwta_k,
            fta_k=self.fta_k,
            fta_bound=self.fta_bound,
        )

    def __call__(self, x):
        return self.V_net(x).squeeze(-1)
