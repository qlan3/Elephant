from collections import namedtuple

import numpy as np
from scipy.stats import bootstrap

from utils.helper import set_one_thread
from utils.plotter import Plotter
from utils.sweeper import unfinished_index


def get_process_result_dict(result, config_idx, mode="Train"):
    result_dict = {
        "Task": result["Task"][0],
        "Model": result["Model"][0],
        "Config Index": config_idx,
        "Perf (mean)": result["Perf"][-10:].mean()
        if mode == "Train"
        else result["Perf"][-5:].mean(),
    }
    return result_dict


def get_csv_result_dict(result, config_idx, mode="Train", ci=95, method="percentile"):
    perf_mean = result["Perf (mean)"].values.tolist()
    if len(perf_mean) > 1:
        CI = bootstrap(
            (perf_mean,), np.mean, confidence_level=ci / 100, method=method
        ).confidence_interval
    else:
        CI = namedtuple("ConfidenceInterval", ["low", "high"])(
            low=perf_mean[0], high=perf_mean[0]
        )
    result_dict = {
        "Task": result["Task"][0],
        "Model": result["Model"][0],
        "Config Index": config_idx,
        "Perf (bmean)": (CI.high + CI.low) / 2,
        f"Perf (ci={ci})": (CI.high - CI.low) / 2,
        "Perf (mean)": result["Perf (mean)"].mean(),
        "Perf (se)": result["Perf (mean)"].sem(ddof=0),
    }
    return result_dict


cfg = {
    "exp": "exp_name",
    "merged": True,
    "x_label": "Step",
    "y_label": "Perf",
    "hue_label": "Model",
    "show": False,
    "imgType": "png",
    "estimator": "mean",
    "ci": "se",
    "x_format": None,
    "y_format": None,
    "xlim": {"min": None, "max": None},
    "ylim": {"min": None, "max": None},
    "EMA": True,
    "loc": "upper left",
    "sweep_keys": ["agent/model_cfg/hidden_act"],
    "sort_by": ["Perf (bmean)"],
    "ascending": [False],
    "runs": 1,
}


def analyze(exp, runs=1):
    set_one_thread()
    cfg["exp"] = exp
    cfg["runs"] = runs
    plotter = Plotter(cfg)

    plotter.csv_merged_results("Test", get_csv_result_dict, get_process_result_dict)
    plotter.plot_results("Test", indexes="all")


if __name__ == "__main__":
    runs = 5
    for exp in ["atari_dqn", "atari_rainbow"]:
        unfinished_index(exp, runs=runs)
        analyze(exp, runs=runs)
