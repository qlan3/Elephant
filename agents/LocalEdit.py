import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import networks
import seaborn as sns
from flax.training.train_state import TrainState
from jax import lax
from utils.helper import load_model_param, rss_memory_usage

from agents import SLTask

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


class LocalEdit(SLTask):
    """
    Edit function value locally
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.task == "EditSin", "Only task EditSin is supported."
        self.output_dim = cfg["agent"]["model_cfg"].get("output_dim", 10)

    def createNN(self, model_name, model_cfg):
        NN = getattr(networks, model_name)(**model_cfg)
        return NN

    @partial(jax.jit, static_argnums=0)
    def update_state(self, state, grads):
        return state.apply_gradients(grads=grads)

    @partial(jax.jit, static_argnums=0)
    def compute_loss(self, params, state, batch):
        pred_y = state.apply_fn(params, batch["x"])["out"]
        loss = jnp.mean(jnp.square(pred_y - batch["y"]))
        perf = lax.stop_gradient(loss)
        return loss, perf

    def run(self):
        self.start_time = time.time()
        self.logger.info("Create train state ...")
        self.logger.info("Create train state: build neural network model")
        model = self.createNN(self.model, self.cfg["agent"]["model_cfg"])
        # Read old params
        self.old_params = load_model_param(self.cfg["agent"]["old_param_path"])
        self.logger.info("Create train state: set optimzer")
        optim = self.set_optim(self.cfg["optim"]["name"], self.cfg["optim"]["kwargs"])
        # Load old params
        self.state = TrainState.create(
            apply_fn=model.apply,
            params=jax.tree_map(lambda x: x, self.old_params),
            tx=optim,
        )
        self.logger.info("Start training and evaluation ...")
        self.train_and_test()
        self.end_time = time.time()
        self.logger.info(f"Memory usage: {rss_memory_usage():.2f} MB")
        self.logger.info(
            f"Time elapsed: {(self.end_time - self.start_time) / 60:.2f} minutes"
        )

    def train_and_test(self):
        batch = {"x": jnp.array([[1.5]]), "y": jnp.array([[-1.5]])}
        for epoch in range(1, self.cfg["epochs"] + 1):
            # Forward: compute loss, performance, and gradient
            (loss, perf), grads = jax.value_and_grad(self.compute_loss, has_aux=True)(
                self.state.params, self.state, batch
            )
            # Backward: update train state
            self.state = self.update_state(self.state, grads)
        # Plot
        self.plot_model(self.old_params, self.state.params, x_t=batch["x"][0, 0])

    def plot_model(self, old_params, new_params, x_t, imgType="png"):
        low, high = -1, 1
        fig, ax = plt.subplots()
        x = jnp.linspace(0, 2, 1000).reshape((-1, 1))
        y_old = self.state.apply_fn(old_params, x)["out"].reshape(-1)
        y_new = self.state.apply_fn(new_params, x)["out"].reshape(-1)
        x = x.reshape((-1))
        plt.plot(x, y_old, color="black", label="Old learned func", linewidth=1)
        plt.plot(x, y_new, color="tab:blue", label="New learned func", linewidth=2)
        plt.scatter(
            x=1.5, y=-1.0, c="tab:orange", label="Old prediction", marker="*", s=100
        )
        plt.scatter(
            x=1.5, y=-1.5, c="tab:red", label="New prediction", marker="^", s=100
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
        ax.get_figure().savefig(self.cfg["logs_dir"] + f"change.{imgType}")
        plt.clf()  # clear figure
        plt.cla()  # clear axis
        plt.close()  # close window
