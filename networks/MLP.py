from typing import Sequence

import flax.linen as nn

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


class MLP(nn.Module):
    """(Elephant) multilayer perceptron."""

    hidden_dims: Sequence[int] = (1000,)
    output_dim: int = 10
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
        self.feat_net = nn.Sequential(layers)
        # Set output layer
        self.out_layer = nn.Dense(
            self.output_dim, kernel_init=self.kernel_init(), use_bias=False
        )

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        feat = self.feat_net(x)
        out = self.out_layer(feat)
        return dict(feat=feat, out=out)
