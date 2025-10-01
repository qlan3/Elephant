"""
Copyright 2021 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License")
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from utils.helper import make_dir

# Set theme
sns.set(style="ticks")
sns.set_context("notebook")
# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
plt.rcParams["ytick.right"] = True
# Set font family, bold, and font size
font = {"family": "sans-serif", "weight": "normal", "size": 16}
matplotlib.rc("font", **font)


GAME_NAMES = [
    ("alien", "Alien"),
    ("amidar", "Amidar"),
    ("assault", "Assault"),
    ("asterix", "Asterix"),
    ("asteroids", "Asteroids"),
    ("atlantis", "Atlantis"),
    ("bank_heist", "Bank Heist"),
    ("battle_zone", "Battlezone"),
    ("beam_rider", "Beam Rider"),
    ("berzerk", "Berzerk"),
    ("bowling", "Bowling"),
    ("boxing", "Boxing"),
    ("breakout", "Breakout"),
    ("centipede", "Centipede"),
    ("chopper_command", "Chopper Command"),
    ("crazy_climber", "Crazy Climber"),
    ("defender", "Defender"),
    ("demon_attack", "Demon Attack"),
    ("double_dunk", "Double Dunk"),
    ("enduro", "Enduro"),
    ("fishing_derby", "Fishing Derby"),
    ("freeway", "Freeway"),
    ("frostbite", "Frostbite"),
    ("gopher", "Gopher"),
    ("gravitar", "Gravitar"),
    ("hero", "H.E.R.O."),
    ("ice_hockey", "Ice Hockey"),
    ("jamesbond", "James Bond 007"),
    ("kangaroo", "Kangaroo"),
    ("krull", "Krull"),
    ("kung_fu_master", "Kung-Fu Master"),
    ("montezuma_revenge", "Montezuma’s Revenge"),
    ("ms_pacman", "Ms. Pac-Man"),
    ("name_this_game", "Name This Game"),
    ("phoenix", "Phoenix"),
    ("pitfall", "Pitfall!"),
    ("pong", "Pong"),
    ("private_eye", "Private Eye"),
    ("qbert", "Q*bert"),
    ("riverraid", "River Raid"),
    ("road_runner", "Road Runner"),
    ("robotank", "Robotank"),
    ("seaquest", "Seaquest"),
    ("skiing", "Skiing"),
    ("solaris", "Solaris"),
    ("space_invaders", "Space Invaders"),
    ("star_gunner", "Stargunner"),
    ("surround", "Surround"),
    ("tennis", "Tennis"),
    ("time_pilot", "Time Pilot"),
    ("tutankham", "Tutankham"),
    ("up_n_down", "Up’n Down"),
    ("venture", "Venture"),
    ("video_pinball", "Video Pinball"),
    ("wizard_of_wor", "Wizard of Wor"),
    ("yars_revenge", "Yars’ Revenge"),
    ("zaxxon", "Zaxxon"),
]
GAME_NAME_MAP = dict(GAME_NAMES)
selected_games = [
    "amidar",
    "battle_zone",
    "bowling",
    "double_dunk",
    "frostbite",
    "kung_fu_master",
    "name_this_game",
    "phoenix",
    "qbert",
    "riverraid",
]


def load_experiment_data_from_results_csv_dir(experiment_details, results_dir):
    df_exps = []
    for ed in experiment_details:
        csv_file = os.path.join(results_dir, ed["agent_id"] + ".csv")
        with open(csv_file, "r") as f:
            df = pd.read_csv(f, index_col=0)
        df = df.assign(agent_id=ed["agent_id"], agent_name=ed["agent_name"])
        # Cut into 50M frames
        df = df[df["frame"] <= 50e6]
        # Select games
        df = df[df["environment_name"].isin(selected_games)]
        df_exps.append(df)
    df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)
    return df_exp


def moving_average(values, window_size):
    # numpy.convolve uses zero for initial missing values, so is not suitable.
    numerator = np.nancumsum(values)
    # The sum of the last window_size values.
    numerator[window_size:] = numerator[window_size:] - numerator[:-window_size]
    denominator = np.ones(len(values)) * window_size
    denominator[:window_size] = np.arange(1, window_size + 1)
    smoothed = numerator / denominator
    assert values.shape == smoothed.shape
    return smoothed


def smooth(df, smoothing_window, index_columns, columns):
    dfg = df.groupby(index_columns)
    for col in columns:
        df[col] = dfg[col].transform(
            lambda s: moving_average(s.values, smoothing_window)
        )
    return df


def environment_pretty(row):
    return GAME_NAME_MAP[row["environment_name"]]


def add_columns(df):
    df["environment"] = df.apply(environment_pretty, axis=1)
    df["frame_millions"] = df["frame"] / int(1e6)
    return df


def smooth_dataframe(df):
    return smooth(
        df,
        smoothing_window=20,
        index_columns=["agent_id", "environment_name", "seed"],
        columns=[
            "normalized_return",
            "eval_episode_return",
        ],
    )


def make_agent_hue_kws(experiments):
    pairs = [(exp["agent_name"], exp["color"]) for exp in experiments]
    agent_names, colors = zip(*pairs)
    hue_kws = dict(color=colors)
    return list(agent_names), hue_kws


# Plot
def plot_individual(df, agent_names, hue_kws):
    g = sns.FacetGrid(
        df.query("agent_name == %s" % agent_names),
        row=None,
        col="environment",
        hue="agent_name",
        height=2.5,
        aspect=1.35,
        col_wrap=2,
        hue_order=agent_names,
        sharey=False,
        hue_kws=hue_kws,
    )

    g = g.map(
        sns.lineplot,
        "frame_millions",
        "eval_episode_return",
        errorbar="se",
        estimator="mean",
        err_kws={"alpha": 0.3},
        linewidth=2,
    )
    g.despine(left=False, top=True, right=False, bottom=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", fontsize=16)
    g.set_axis_labels("Frame (millions)", "")

    # Create legend from the final axes.
    legend_x_margin = 0.2
    legend_y_offset_inches = 0
    legend_y_offset = legend_y_offset_inches / g.figure.get_figheight()
    g.axes[-1].legend(
        bbox_to_anchor=(legend_x_margin, legend_y_offset, 1 - 2 * legend_x_margin, 0),
        bbox_transform=g.figure.transFigure,
        mode="expand",
        ncol=2,
        fontsize=16,
        borderaxespad=0,
        loc="lower left",
        frameon=False,
    )
    g.figure.subplots_adjust(bottom=0.125)

    for ax in g.axes:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    return g


def plot_summary(df, agent_names, hue_kws):
    sns.set_palette(sns.color_palette("tab10", n_colors=len(agent_names)))
    df = df.query("agent_name == %s" % agent_names)
    df = (
        df.groupby(["agent_name", "frame", "seed"])
        .median(numeric_only=True)
        .reset_index()
    )
    df["normalized_return_percent"] = df["normalized_return"] * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        x=df["frame_millions"],
        y=df["normalized_return_percent"],
        hue=df["agent_name"],
        errorbar="se",
        estimator="mean",
        err_kws={"alpha": 0.3},
        linewidth=2,
        hue_order=agent_names,
        palette=hue_kws["color"],
        legend="brief",
        ax=ax,
    )
    ax.set_title("")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.set_xlabel("Frame (millions)", fontsize=16)
    ax.set_ylabel("Human-normalized score", fontsize=16)
    ax.set(ylim=(-5, None))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    # Grab generated legend and remove label 'agent'.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        loc="upper left",
        fontsize=16,
        handles=handles,
        labels=labels,
        frameon=False,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    make_dir("./figures/")
    experiments = {
        "dqn": [
            dict(agent_id="Atari_DQN_ReLU", agent_name="ReLU", color="tab:red"),
            dict(agent_id="Atari_DQN_Tanh", agent_name="Tanh", color="tab:cyan"),
            dict(agent_id="Atari_DQN_Maxout", agent_name="Maxout", color="tab:blue"),
            dict(agent_id="Atari_DQN_LWTA", agent_name="LWTA", color="tab:orange"),
            dict(agent_id="Atari_DQN_FTA", agent_name="FTA", color="tab:green"),
            dict(agent_id="Atari_DQN_Elephant", agent_name="Elephant", color="black"),
        ],
        "rainbow": [
            dict(agent_id="Atari_Rainbow_ReLU", agent_name="ReLU", color="tab:red"),
            dict(agent_id="Atari_Rainbow_Tanh", agent_name="Tanh", color="tab:cyan"),
            dict(
                agent_id="Atari_Rainbow_Maxout", agent_name="Maxout", color="tab:blue"
            ),
            dict(agent_id="Atari_Rainbow_LWTA", agent_name="LWTA", color="tab:orange"),
            dict(agent_id="Atari_Rainbow_FTA", agent_name="FTA", color="tab:green"),
            dict(
                agent_id="Atari_Rainbow_Elephant", agent_name="Elephant", color="black"
            ),
        ],
    }

    for algo in ["dqn", "rainbow"]:
        df_exp_raw = load_experiment_data_from_results_csv_dir(
            experiments[algo], "results/"
        )
        df_exp = df_exp_raw.pipe(add_columns).pipe(smooth_dataframe)
        df = df_exp.sort_values(by=["agent_id", "environment_name", "seed", "frame"])

        fig = plot_summary(df, *make_agent_hue_kws(experiments[algo]))
        fig.savefig(f"./figures/atari_{algo}_summary.png")
        plt.close(fig)

        g = plot_individual(df, *make_agent_hue_kws(experiments[algo]))
        g.figure.savefig(f"./figures/atari_{algo}_individual.png")
        plt.close(g.figure)
