import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from agents import Regression
from utils.helper import tree_transpose

# Set theme
sns.set(style="ticks")
sns.set_context("paper")
# Set font family, bold, and font size
font = {"size": 16}
matplotlib.rc("font", **font)
# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0


class StreamRegression(Regression):
    """
    Stream regression task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def train_and_test(self):
        if self.task in ["StreamSin", "StreamSinPlus"]:
            self.vmap_grad_fn = jax.vmap(self.grad_fn, in_axes=(None, 0), out_axes=0)
        """Train for a single pass."""
        mode = "Train"
        data_size = len(self.data[mode]["x"])
        batch_num = data_size // self.cfg["batch_size"]
        perms = jnp.array(range(batch_num * self.cfg["batch_size"]))
        perms = perms.reshape((batch_num, self.cfg["batch_size"]))
        if self.task in ["StreamSin", "StreamSinPlus"]:
            super().plot_model(0, self.state.params, x_t=None)
        for i, perm in enumerate(perms):
            batch_start_time = time.time()
            progress = (i + 1) / batch_num
            batch = {
                "x": self.data[mode]["x"][perm, ...],
                "y": self.data[mode]["y"][perm, ...],
            }
            logs = []
            # Clone neural network params
            self.old_params = jax.tree_map(lambda x: x, self.state.params)
            for _ in range(self.cfg["epochs"]):
                # Train for multiple epochs for a batch
                self.state, log = self.train_batch(self.state, batch)
                logs.append(log)
            # Log
            logs = tree_transpose(logs)
            logs = jax.device_get(logs)
            batch_loss = np.mean(logs["loss"])
            batch_perf = np.mean(logs["perf"])
            # Check NaN error
            if np.isnan(batch_loss) or np.isnan(batch_perf):
                self.logger.info("NaN error detected!")
                break
            # Evaluation
            test_isnan = self.evaluate_epoch(
                i + 1, progress, self.cfg["batch_size"], key="Batch"
            )
            # Check NaN error
            if test_isnan:
                self.logger.info("NaN error detected!")
                break
            # Display
            if ((i + 1) % self.cfg["display_interval"] == 0) or (progress == 1.0):
                speed = time.time() - batch_start_time
                eta = (batch_num - i - 1) * speed / 60 if speed > 0 else -1
                self.logger.info(
                    f"<{self.config_idx}> {self.task} {self.model} Speed={speed:.2f} (s/batch), ETA={eta:.2f} (mins)"
                )
                if self.task in ["StreamSin", "StreamSinPlus"]:
                    # Plot: true func, learned func, normalized NTK
                    self.plot_model(
                        i + 1, self.old_params, self.state.params, x_t=batch["x"]
                    )

    @partial(jax.jit, static_argnums=0)
    def train_batch(self, state, batch):
        # Forward: compute loss, performance, and gradient
        (loss, perf), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
            state.params, state, batch
        )
        # Backward: update train state
        state = self.update_state(state, grads)
        log = dict(loss=loss, perf=perf)
        return state, log

    @partial(jax.jit, static_argnums=0)
    def grad_fn(self, params, x):
        def pred_y(params, x):
            return jnp.mean(self.state.apply_fn(params, x[None,])["out"])

        grad = jax.grad(pred_y)(params, x)
        return self.pytree2array(grad)

    def plot_model(self, epoch, old_params, new_params, x_t, imgType="png"):
        if self.task in ["SinPlus", "StreamSinPlus"]:
            low, high = 0, 1
        elif self.task in ["Sin", "StreamSin"]:
            low, high = -1, 1
        fig, ax = plt.subplots()
        x = jnp.linspace(0, 2, 1000).reshape((-1, 1))
        y_true = self.y_true_fn(x)
        y_pred = self.state.apply_fn(new_params, x)["out"].reshape(-1)
        # Compute the dot product of gradient and gradient_t
        g_t = self.vmap_grad_fn(old_params, x_t)[0]
        g = self.vmap_grad_fn(old_params, x)
        ntk = jnp.dot(g, g_t)
        norm_ntk = ntk / jnp.abs(ntk).max()
        x = x.reshape((-1))
        plt.plot(x, y_true, color="black", label="True func", linewidth=1)
        plt.plot(x, y_pred, color="tab:blue", label="Learned func", linewidth=2)
        plt.plot(x, norm_ntk, color="tab:red", label="Normalized NTK", linewidth=2)
        plt.vlines(
            x_t,
            2 * low,
            2 * high,
            colors="tab:orange",
            linestyles="dashed",
            label="$x_t$",
            linewidth=2,
        )
        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("y", fontsize=18)
        ax.locator_params(nbins=4, axis="y")
        plt.yticks(size=18)
        plt.xticks(size=18)
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(2 * low, 2 * high)
        ax.legend(loc="best", frameon=False, fontsize=18)
        plt.tight_layout()
        ax.get_figure().savefig(self.cfg["logs_dir"] + f"{epoch}.{imgType}")
        plt.clf()  # clear figure
        plt.cla()  # clear axis
        plt.close()  # close window
