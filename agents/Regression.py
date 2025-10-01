import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import lax
from utils.helper import save_model_param

from agents import Classification

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


class Regression(Classification):
    """
    Regression task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.task in ["Sin", "StreamSin"]:
            self.y_true_fn = lambda x: jnp.sin(x * jnp.pi)
        elif self.task in ["SinPlus", "StreamSinPlus"]:
            self.y_true_fn = (
                lambda x: 0.1 * jnp.sin(10.0 * (x - 1.0)) + jnp.square(x - 1.0) + 0.1
            )

    @partial(jax.jit, static_argnums=0)
    def compute_loss(self, params, state, batch):
        pred_y = state.apply_fn(params, batch["x"])["out"]
        loss = jnp.mean(jnp.square(pred_y - batch["y"]))
        perf = lax.stop_gradient(loss)
        return loss, perf

    @partial(jax.jit, static_argnums=0)
    def compute_SKL_loss(self, params, state, batch):
        """Compute Set KL loss + original loss"""
        outs = state.apply_fn(params, batch["x"])
        pred_y, feat = outs["out"], outs["feat"]
        # Compute orignal loss
        loss = jnp.mean(jnp.square(pred_y - batch["y"]))
        perf = lax.stop_gradient(loss)
        # Compute SKL loss
        feat = feat.mean(axis=0) + 1e-8
        skl_loss = jnp.where(
            lax.stop_gradient(feat > self.cfg["srnn_beta"]),
            self.cfg["srnn_beta"] / feat - jnp.log(self.cfg["srnn_beta"] / feat) - 1.0,
            0.0,
        )
        # Compute total loss
        loss += self.cfg["srnn_lambda"] * skl_loss.sum()
        return loss, perf

    def evaluate_epoch(self, epoch, progress, batch_size, mode="Test", key="Epoch"):
        assert mode in ["Test", "Valid"]
        # Forward
        epoch_loss, epoch_perf, is_dead = self.evaluate_batch(
            self.state, self.data[mode]
        )
        epoch_loss = float(jax.device_get(epoch_loss))
        epoch_perf = float(jax.device_get(epoch_perf))
        # Check NaN error
        if np.isnan(epoch_loss) or np.isnan(epoch_perf):
            return True
        # Compute the number of active units
        data_size = len(self.data[mode]["x"])
        active_unit_num = (is_dead <= (1 - self.percent_eps) * data_size).sum()
        active_unit_num = int(jax.device_get(active_unit_num))
        # Save evaluation result
        self.save_results(
            mode,
            epoch,
            progress,
            epoch_loss,
            epoch_perf,
            key,
            active_unit_num=active_unit_num,
        )
        return False

    @partial(jax.jit, static_argnums=0)
    def evaluate_batch(self, state, batch):
        preds = state.apply_fn(state.params, batch["x"])
        pred_y, feat = preds["out"], preds["feat"]
        loss = jnp.mean(jnp.square(pred_y - batch["y"]))
        perf = lax.stop_gradient(loss)
        # Count number of active units
        is_dead = jnp.abs(lax.stop_gradient(feat)) < self.value_eps
        is_dead = jnp.sum(is_dead, axis=0)
        return loss, perf, is_dead

    def plot_model(self, epoch, params, x_t=None, imgType="png"):
        if self.task in ["SinPlus", "StreamSinPlus"]:
            low, high = 0, 1
        elif self.task in ["Sin", "StreamSin"]:
            low, high = -1, 1
        fig, ax = plt.subplots()
        x = jnp.linspace(0, 2, 1000)
        y_true = self.y_true_fn(x)
        y_pred = self.state.apply_fn(params, x.reshape((-1, 1)))["out"].reshape(-1)
        plt.plot(x, y_true, color="black", label="True func", linewidth=1)
        plt.plot(x, y_pred, color="tab:blue", label="Learned func", linewidth=2)
        if x_t is not None:
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
        plt.xticks(size=18)
        plt.yticks(size=18)
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(2 * low, 2 * high)
        ax.legend(loc="best", frameon=False, fontsize=18)
        plt.tight_layout()
        ax.get_figure().savefig(self.cfg["logs_dir"] + f"{epoch}.{imgType}")
        plt.clf()  # clear figure
        plt.cla()  # clear axis
        plt.close()  # close window

    def train_and_test(self):
        for epoch in range(1, self.cfg["epochs"] + 1):
            epoch_start_time = time.time()
            progress = epoch / self.cfg["epochs"]
            train_isnan = self.train_epoch(epoch, progress, self.cfg["batch_size"])
            test_isnan = self.evaluate_epoch(epoch, progress, self.cfg["batch_size"])
            # Check NaN error
            if train_isnan or test_isnan:
                self.logger.info("NaN error detected!")
                break
            # Display
            if (epoch % self.cfg["display_interval"] == 0) or (progress == 1.0):
                speed = time.time() - epoch_start_time
                eta = (self.cfg["epochs"] - epoch) * speed / 60 if speed > 0 else -1
                self.logger.info(
                    f"<{self.config_idx}> {self.task} {self.model} Speed={speed:.2f} (s/epoch), ETA={eta:.2f} (mins)"
                )
                if self.task in ["Sin", "SinPlus"]:
                    # Plot learned model and true model
                    self.plot_model(epoch, self.state.params)
        # Save model parameters
        save_model_param(self.state.params, self.cfg["model_path"])
