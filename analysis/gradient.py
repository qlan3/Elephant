import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import seaborn as sns

from utils.helper import make_dir

# Set theme
sns.set_theme(style="ticks", context="notebook")
# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
# Set font family, bold, and font size
font = {"family": "sans-serif", "weight": "normal", "size": 16}
matplotlib.rc("font", **font)


def compute_grad_covariance(grads):
    n = grads.shape[0]
    cov = np.zeros((n, n))
    norms = LA.norm(grads, axis=-1)
    for i in range(0, n - 1):
        for j in range(i, n):
            cov[i, j] = np.dot(grads[i], grads[j]) / norms[i] / norms[j]
            cov[j, i] = cov[i, j]
    return cov


def plot_grad_covariance(dir, epochs):
    with open(dir + "config.json", "r") as f:
        cfg = json.load(f)
    task, agent, hidden_act = (
        cfg["task"]["name"][:-14],
        cfg["agent"]["name"],
        cfg["agent"]["model_cfg"]["hidden_act"],
    )
    # Compute and Plot
    n = len(epochs)
    fig, axes = plt.subplots(
        nrows=1, ncols=n, sharex=True, sharey=True, figsize=(n * 4, 4)
    )
    cbar_ax = fig.add_axes([0.915, 0.15, 0.02, 0.7])
    for i, epoch in enumerate(epochs):
        grads = np.load(dir + f"grad_{epoch}.npy")
        cov = compute_grad_covariance(grads)
        sns.heatmap(
            cov,
            ax=axes[i],
            vmin=-1,
            vmax=1,
            square=True,
            center=0,
            cmap="RdBu",
            cbar=(i == n - 1),
            cbar_ax=None if i < n - 1 else cbar_ax,
        )
        axes[i].set_title(f"Frame={epoch * 0.4:.1f}M")
    # Adjust figure layout
    plt.subplots_adjust(left=0.05)
    plt.savefig(f"figures/grad/{agent}_{task}_{hidden_act}.png", bbox_inches="tight")
    plt.clf()  # clear figure
    plt.cla()  # clear axis
    plt.close()  # close window


if __name__ == "__main__":
    make_dir("./figures/grad")
    for i in range(1, 60 + 1):
        plot_grad_covariance(f"logs/grad_atari_dqn/{i}/", [2, 62, 122])
        plot_grad_covariance(f"logs/grad_atari_rainbow/{i}/", [2, 62, 122])
