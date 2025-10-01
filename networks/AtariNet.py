import math
from typing import Any, Sequence

import numpy as np
import torch
from tianshou.utils.net.discrete import NoisyLinear
from torch import Tensor, nn

activations = {
    "ReLU": nn.ReLU(inplace=True),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "LeakyReLU": nn.LeakyReLU(),
}


# From https://github.com/pytorch/pytorch/issues/61292
@torch.jit.script
def linspace(start: Tensor, stop: Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    out = start + steps * (stop - start)
    return out


def linspace_bias_init(layer, scale: float = 0.0):
    with torch.no_grad():
        start = (
            -1.0
            * math.sqrt(3)
            * scale
            * torch.ones_like(layer.bias.data, requires_grad=False)
        )
        end = (
            math.sqrt(3) * scale * torch.ones_like(layer.bias.data, requires_grad=False)
        )
        layer.bias.data = linspace(start, end, num=layer.bias.data.shape[0])
        return layer


class Elephant(nn.Module):
    """Adaptive elementwise elephant activation function."""

    def __init__(
        self, input_shape: int = 1, sigma: float = 1.0, d: int = 2, h: float = 1.0
    ):
        super().__init__()
        self.d = d
        self.log_h = nn.Parameter(
            torch.ones(input_shape, requires_grad=True) * math.log(h)
        )
        self.log_sigma = nn.Parameter(
            torch.ones(input_shape, requires_grad=True) * math.log(sigma)
        )

    def forward(self, x):
        h = torch.exp(self.log_h)
        sigma = torch.exp(self.log_sigma)
        y = h / (1.0 + torch.pow((x / sigma).abs(), self.d))
        return y


class Maxout(nn.Module):
    """Maxout activation function."""

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def forward(self, x):
        *rest, last = x.shape
        assert last % self.k == 0, f"Last dimension must be divisible by k={self.k}"
        y = x.view(*rest, last // self.k, self.k)
        return torch.max(y, dim=-1).values


class LWTA(nn.Module):
    """Winner-take-all activation function."""

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def forward(self, x):
        *rest, last = x.shape
        assert last % self.k == 0, f"Last dimension must be divisible by k={self.k}"
        y = x.view(*rest, last // self.k, self.k)
        idx = torch.argmax(y, dim=-1, keepdim=True)
        one_hot = torch.zeros_like(y).scatter_(-1, idx, 1)
        y = (y * one_hot).view(*rest, last)
        return y


class FTA(nn.Module):
    """Fuzzy tiling activation function."""

    def __init__(self, k: int = 20, bound: float = 20.0):
        super().__init__()
        assert k > 0, "The number of tiles must be positive."
        self.k = k
        self.bound = bound
        self.eta = 2 * bound / k
        self.delta = self.eta
        self.register_buffer(
            "c_mat", torch.linspace(-bound, bound - self.delta, steps=k)
        )

    def relu_sum(self, c, x):
        return torch.relu(c - x) + torch.relu(x - self.delta - c)

    def I_eta_plus(self, x):
        return torch.where(x <= self.eta, x, torch.ones_like(x))

    def forward(self, z):
        z = z.unsqueeze(-1)  # Expand to shape [..., 1]
        out = 1.0 - self.I_eta_plus(self.relu_sum(self.c_mat, z))
        return out.view(*z.shape[:-2], -1)


class AtariDQNNet(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        action_shape: Sequence[int],
        device: str | int | torch.device = "cpu",
        last_hidden_dim: int = 512,
        hidden_act: str = "ReLU",
        # Elephant
        d: int = 4,
        sigma: float = 0.1,
        bias_std: float = 0.0,
        h: float = 1.0,
        # Maxout
        maxout_k: int = -1,
        # LWTA
        lwta_k: int = -1,
        # FTA
        fta_k: int = -1,
        fta_bound: float = -1.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        layers = [
            nn.Conv2d(channel, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        ]
        with torch.no_grad():
            self.feature_dim = int(
                np.prod(
                    nn.Sequential(*layers)(
                        torch.zeros(1, channel, height, width)
                    ).shape[1:]
                )
            )
        if hidden_act == "Elephant":
            layers.append(
                linspace_bias_init(
                    nn.Linear(self.feature_dim, last_hidden_dim), bias_std
                )
            )
            layers.append(nn.LayerNorm(last_hidden_dim))
            layers.append(
                Elephant(input_shape=last_hidden_dim, sigma=sigma, d=d, h=h)
            )  # Elephant activation
        elif hidden_act == "Maxout":
            layers.append(nn.Linear(self.feature_dim, last_hidden_dim * maxout_k))
            layers.append(Maxout(k=maxout_k))
        elif hidden_act == "LWTA":
            assert last_hidden_dim % lwta_k == 0, (
                f"Last hidden_dim={last_hidden_dim} must be divisible by k={lwta_k}."
            )
            layers.append(nn.Linear(self.feature_dim, last_hidden_dim))
            layers.append(LWTA(k=lwta_k))
        elif hidden_act == "FTA":
            assert last_hidden_dim % fta_k == 0, (
                f"Last hidden_dim={last_hidden_dim} must be divisible by k={fta_k}."
            )
            layers.append(nn.Linear(self.feature_dim, last_hidden_dim // fta_k))
            layers.append(FTA(k=fta_k, bound=fta_bound))
        elif hidden_act in activations.keys():
            layers.append(nn.Linear(self.feature_dim, last_hidden_dim))
            layers.append(activations[hidden_act])
        layers.append(nn.Linear(last_hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


# Adapted from https://github.com/thu-ml/tianshou/blob/v0.4.10/examples/atari/atari_rainbow.py
class AtariRainbowNet(nn.Module):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        action_shape: Sequence[int],
        device: str | int | torch.device = "cpu",
        last_hidden_dim: int = 512,
        hidden_act: str = "ReLU",
        num_atoms: int = 51,
        noisy_std: float = 0.1,
        # Elephant
        d: int = 4,
        sigma: float = 0.1,
        bias_std: float = 0.0,  # Ingored due to NoisyLinear
        h: float = 1.0,
        # Maxout
        maxout_k: int = -1,
        # LWTA
        lwta_k: int = -1,
        # FTA
        fta_k: int = -1,
        fta_bound: float = -1.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.num_atoms = num_atoms
        self.action_num = np.prod(action_shape)
        self.output_dim = self.action_num * self.num_atoms
        self.feature_net = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.feature_dim = int(
                np.prod(
                    self.feature_net(torch.zeros(1, channel, height, width)).shape[1:]
                )
            )
        q_layers, v_layers = [], []
        if hidden_act == "Elephant":
            q_layers.append(NoisyLinear(self.feature_dim, last_hidden_dim, noisy_std))
            v_layers.append(NoisyLinear(self.feature_dim, last_hidden_dim, noisy_std))
            # Activation normalization
            q_layers.append(nn.LayerNorm(last_hidden_dim))
            v_layers.append(nn.LayerNorm(last_hidden_dim))
            # Elephant activation
            q_layers.append(
                Elephant(input_shape=last_hidden_dim, sigma=sigma, d=d, h=h)
            )
            v_layers.append(
                Elephant(input_shape=last_hidden_dim, sigma=sigma, d=d, h=h)
            )
        elif hidden_act == "Maxout":
            q_layers.append(
                NoisyLinear(self.feature_dim, last_hidden_dim * maxout_k, noisy_std)
            )
            v_layers.append(
                NoisyLinear(self.feature_dim, last_hidden_dim * maxout_k, noisy_std)
            )
            q_layers.append(Maxout(k=maxout_k))
            v_layers.append(Maxout(k=maxout_k))
        elif hidden_act == "LWTA":
            assert last_hidden_dim % lwta_k == 0, (
                f"Last hidden_dim={last_hidden_dim} must be divisible by k={lwta_k}."
            )
            q_layers.append(NoisyLinear(self.feature_dim, last_hidden_dim, noisy_std))
            v_layers.append(NoisyLinear(self.feature_dim, last_hidden_dim, noisy_std))
            q_layers.append(LWTA(k=lwta_k))
            v_layers.append(LWTA(k=lwta_k))
        elif hidden_act == "FTA":
            assert last_hidden_dim % fta_k == 0, (
                f"Last hidden_dim={last_hidden_dim} must be divisible by k={fta_k}."
            )
            q_layers.append(
                NoisyLinear(self.feature_dim, last_hidden_dim // fta_k, noisy_std)
            )
            v_layers.append(
                NoisyLinear(self.feature_dim, last_hidden_dim // fta_k, noisy_std)
            )
            q_layers.append(FTA(k=fta_k, bound=fta_bound))
            v_layers.append(FTA(k=fta_k, bound=fta_bound))
        elif hidden_act in activations.keys():
            q_layers.append(NoisyLinear(self.feature_dim, last_hidden_dim, noisy_std))
            v_layers.append(NoisyLinear(self.feature_dim, last_hidden_dim, noisy_std))
            q_layers.append(activations[hidden_act])
            v_layers.append(activations[hidden_act])
        q_layers.append(
            NoisyLinear(last_hidden_dim, self.action_num * self.num_atoms, noisy_std)
        )
        v_layers.append(NoisyLinear(last_hidden_dim, self.num_atoms, noisy_std))
        self.Q = nn.Sequential(*q_layers)
        self.V = nn.Sequential(*v_layers)

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        phi = self.feature_net(obs)
        q = self.Q(phi).view(-1, self.action_num, self.num_atoms)
        v = self.V(phi).view(-1, 1, self.num_atoms)
        logits = q - q.mean(dim=1, keepdim=True) + v
        probs = logits.softmax(dim=2)
        return probs, state
