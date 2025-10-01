import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import lax, random, tree_util

from utils.helper import set_random_seed
from utils.logger import Logger


class SLTask:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        # Set default device
        try:
            if not (len(jax.devices(backend="cuda")) > 0 and "cuda" in cfg["device"]):
                self.cfg["device"] = "cpu"
        except Exception:
            self.cfg["device"] = "cpu"
        if self.cfg["device"] != "cuda":
            if self.cfg["device"] == "cpu":
                jax.config.update("jax_platform_name", "cpu")
            elif "cuda" in self.cfg["device"]:
                backend, idx = self.cfg["device"].split(":")
                device = jax.devices(backend=backend)[int(idx)]
                jax.config.update("jax_default_device", device)
        try:
            self.task = cfg["task"]["name"]
        except Exception:
            self.task = cfg["task"]
        self.model = self.cfg["agent"]["model_name"]
        self.config_idx = cfg["config_idx"]
        self.logger = Logger(cfg["logs_dir"])
        if self.cfg["generate_random_seed"]:
            self.cfg["seed"] = np.random.randint(int(1e6))
        self.seed = random.PRNGKey(self.cfg["seed"])
        set_random_seed(self.cfg["seed"])
        self.cfg_path = self.cfg["cfg_path"]
        self.log_path = {
            "Train": self.cfg["train_log_path"],
            "Test": self.cfg["test_log_path"],
        }
        self.results = {"Train": [], "Test": []}
        self.save_config()
        # Get available cores
        total_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        self.logger.info(f"Total device: {total_device_count}")
        self.logger.info(f"Local device: {local_device_count}")

    def createNN(self, model_name, model_cfg):
        raise NotImplementedError

    def set_optim(self, optim_name, optim_kwargs, schedule=None):
        optim_kwargs.setdefault("anneal_lr", False)
        optim_kwargs.setdefault("grad_clip", -1)
        optim_kwargs.setdefault("max_grad_norm", -1)
        anneal_lr = optim_kwargs["anneal_lr"]
        grad_clip = optim_kwargs["grad_clip"]
        max_grad_norm = optim_kwargs["max_grad_norm"]
        del (
            optim_kwargs["anneal_lr"],
            optim_kwargs["grad_clip"],
            optim_kwargs["max_grad_norm"],
        )
        assert not (grad_clip > 0 and max_grad_norm > 0), (
            "Cannot apply both grad_clip and max_grad_norm at the same time."
        )
        if anneal_lr and schedule is not None:
            optim_kwargs["learning_rate"] = schedule
        if grad_clip > 0:
            optim = optax.chain(
                optax.clip(grad_clip),
                getattr(optax, optim_name.lower())(**optim_kwargs),
            )
        elif max_grad_norm > 0:
            optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                getattr(optax, optim_name.lower())(**optim_kwargs),
            )
        else:
            optim = getattr(optax, optim_name.lower())(**optim_kwargs)
        return optim

    def run(self):
        raise NotImplementedError

    def pytree2array(self, values):
        leaves = tree_util.tree_leaves(lax.stop_gradient(values))
        a = jnp.concatenate(leaves, axis=None)
        return a

    def save_config(self):
        cfg_json = json.dumps(self.cfg, indent=2)
        f = open(self.cfg_path, "w")
        f.write(cfg_json)
        f.close()

    def save_to_file(self, mode):
        results = pd.DataFrame(self.results[mode])
        results["Task"] = results["Task"].astype("category")
        results["Model"] = results["Model"].astype("category")
        results.to_feather(self.log_path[mode])
