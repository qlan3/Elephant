from typing import Sequence

import flax.linen as nn

from networks.MLP import MLP
from networks.utils import (
    Initializer,
    activations,
    linspace_bias_init,
    torch_kernel_init,
)


class CNN(nn.Module):
    """CNN + (elephant) multilayer perceptron."""

    # CNN cfg
    cnn_hidden_dim: int = 64
    cnn_hidden_act: str = "ReLU"
    # MLP cfg
    mlp_hidden_dims: Sequence[int] = (1000,)
    mlp_hidden_act: str = "ReLU"
    mlp_d: int = 4
    mlp_sigma: float = 0.05
    mlp_bias_std: float = 0.0
    mlp_input_standardize: bool = False
    mlp_act_standardize: bool = False
    mlp_input_layernorm: bool = False
    mlp_act_layernorm: bool = False
    # Others
    output_dim: int = 10
    kernel_init: Initializer = torch_kernel_init()
    bias_init: Initializer = linspace_bias_init

    def setup(self):
        # Set CNN feature network
        layers = [
            nn.Conv(
                self.cnn_hidden_dim,
                kernel_size=(3, 3),
                kernel_init=self.kernel_init,
                use_bias=True,
            ),
            activations[self.cnn_hidden_act],
            lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)),
        ]
        self.cnn = nn.Sequential(layers)
        # Set MLP output network
        if self.mlp_hidden_act == "Elephant":
            self.mlp = MLP(
                hidden_dims=self.mlp_hidden_dims,
                output_dim=self.output_dim,
                hidden_act=self.mlp_hidden_act,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                d=self.mlp_d,
                sigma=self.mlp_sigma,
                bias_std=self.mlp_bias_std,
            )
        else:
            self.mlp = MLP(
                hidden_dims=self.mlp_hidden_dims,
                output_dim=self.output_dim,
                hidden_act=self.mlp_hidden_act,
                kernel_init=self.kernel_init,
            )

    def __call__(self, x):
        feat = self.cnn(x)
        out = self.mlp(feat)["out"]
        return dict(feat=feat, out=out)
