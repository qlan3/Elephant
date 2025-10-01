import argparse
import sys

import agents
from utils.helper import make_dir
from utils.sweeper import Sweeper


def main(argv):
    parser = argparse.ArgumentParser(description="Config file")
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configs/mnist.json",
        help="Configuration file for the chosen model",
    )
    parser.add_argument("--config_idx", type=int, default=1, help="Configuration index")
    args = parser.parse_args()

    sweeper = Sweeper(args.config_file)
    cfg = sweeper.generate_config_for_idx(args.config_idx)

    # Set config dict default value
    cfg.setdefault("show_tb", False)
    cfg.setdefault("save_model", False)
    cfg.setdefault("show_progress", False)
    cfg.setdefault("resume_from_log", False)
    cfg.setdefault("save_interval", 1)
    cfg.setdefault("srnn_lambda", -1)
    cfg.setdefault("srnn_beta", -1)

    # Set experiment name and log paths
    cfg["exp"] = args.config_file.split("/")[-1].split(".")[0]
    local_logs_dir = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
    make_dir(local_logs_dir)
    cfg["logs_dir"] = local_logs_dir
    cfg["train_log_path"] = cfg["logs_dir"] + "result_Train.feather"
    cfg["test_log_path"] = cfg["logs_dir"] + "result_Test.feather"
    cfg["model_path"] = cfg["logs_dir"] + "model.pickle"
    cfg["cfg_path"] = cfg["logs_dir"] + "config.json"

    if "NoFrameskip" in cfg["task"]["name"]:
        cfg["model_path"] = cfg["logs_dir"] + "model.pth"
        cfg["ckpt_path"] = cfg["logs_dir"] + "ckpt.pth"
        if "Breakout" in cfg["task"]["name"]:  # Special case for BreakoutNoFrameskip-v4
            cfg["n_step"] = 1
    exp = getattr(agents, cfg["agent"]["name"])(cfg)
    exp.run()


if __name__ == "__main__":
    main(sys.argv)
