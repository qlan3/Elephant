from utils.plotter import Plotter
from utils.sweeper import memory_info, time_info, unfinished_index


def get_process_result_dict(result, config_idx, mode="Train"):
    result_dict = {
        "Task": result["Task"][0],
        "Model": result["Model"][0],
        "Config Index": config_idx,
        "Perf (mean)": result["Perf"][-5:].mean(skipna=False),
    }
    return result_dict


def get_csv_result_dict(result, config_idx, mode="Train"):
    result_dict = {
        "Task": result["Task"][0],
        "Model": result["Model"][0],
        "Config Index": config_idx,
        "Perf (mean)": result["Perf (mean)"].mean(skipna=False),
        "Perf (se)": result["Perf (mean)"].sem(ddof=0),
    }
    return result_dict


cfg = {
    "exp": "exp_name",
    "merged": True,
    "x_label": "Batch",
    "y_label": "Perf",
    "rolling_score_window": -1,
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
    "loc": "lower right",
    "sweep_keys": ["agent/model_cfg/hidden_act"],
    "sort_by": ["Perf (mean)", "Perf (se)"],
    "ascending": [True, True],
    "runs": 1,
}


def analyze(exp, runs=1):
    cfg["exp"] = exp
    cfg["runs"] = runs
    plotter = Plotter(cfg)

    plotter.csv_results("Test", get_csv_result_dict, get_process_result_dict)
    plotter.plot_results(mode="Test", indexes="all")


if __name__ == "__main__":
    runs = 5
    exp_list = ["streamsin"]
    for exp in exp_list:
        unfinished_index(exp, runs=runs)
        memory_info(exp, runs=runs)
        time_info(exp, runs=runs)
        analyze(exp, runs=runs)
