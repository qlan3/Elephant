import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tianshou.data import Collector, VectorReplayBuffer

import networks
from utils.env import make_atari_env
from utils.helper import rss_memory_usage, set_one_thread, set_random_seed
from utils.logger import AtariLogger
from utils.policy import DQNPolicy
from utils.trainer import OffpolicyTrainerGradCovariance as OffpolicyTrainer


class AtariDQN(object):
    """
    Implementation of DQN for Atari games.
    """

    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.setdefault("compute_grad_covariance_at_epochs", None)
        try:
            self.task = cfg["task"]["name"]
        except:  # noqa: E722
            self.task = cfg["task"]
        self.model_name = self.cfg["agent"]["model_name"]
        if torch.cuda.is_available() and "cuda" in cfg["device"]:
            self.device = cfg["device"]
        else:
            self.cfg["device"] = "cpu"
            self.device = "cpu"
        self.config_idx = cfg["config_idx"]
        if self.cfg["generate_random_seed"]:
            self.cfg["seed"] = np.random.randint(int(1e6))
        self.model_path = self.cfg["model_path"]
        self.cfg_path = self.cfg["cfg_path"]
        self.save_config()
        self.discount = cfg["discount"]
        self.batch_size = cfg["batch_size"]
        self.cfg["epoch"] = int(self.cfg["epoch"])
        self.save_only_last_obs = True
        if self.cfg["step_per_collect"] < 0:
            self.cfg["step_per_collect"] = round(1 / self.cfg["update_per_step"])
        # Make envs
        self.envs = dict()
        self.env, self.envs["Train"], self.envs["Test"] = make_atari_env(
            task=self.task,
            seed=self.cfg["seed"],
            training_num=self.cfg["task"]["train_num"],
            test_num=self.cfg["task"]["test_num"],
            scale=self.cfg["task"]["scale_obs"],
            frame_stack=4,
        )
        self.state_shape = self.get_state_shape(self.env)
        self.action_shape = self.get_action_shape(self.env)
        self.log_path = {
            "Train": self.cfg["train_log_path"],
            "Test": self.cfg["test_log_path"],
        }
        # Set python logger and tensorboard logger
        self.logger = AtariLogger(
            cfg["logs_dir"], save_interval=self.cfg["save_interval"]
        )
        # Create Q network
        self.net = self.createNN(self.model_name, self.cfg["agent"]["model_cfg"])
        # Set optimizer
        self.optimizer = getattr(torch.optim, self.cfg["optim"]["name"])(
            self.net.parameters(), **self.cfg["optim"]["kwargs"]
        )
        # Set replay buffer: `save_last_obs` and `stack_num` can be removed when you have enough RAM
        self.buffer = VectorReplayBuffer(
            total_size=self.cfg["buffer_size"],
            buffer_num=self.cfg["task"]["train_num"],
            ignore_obs_next=True,
            save_only_last_obs=self.save_only_last_obs,
            stack_num=self.cfg["frames_stack"],
        )
        # Define policy
        self.policy = DQNPolicy(
            model=self.net,
            optim=self.optimizer,
            discount_factor=self.discount,
            estimation_step=self.cfg["n_step"],
            target_update_freq=self.cfg["target_update_steps"],
            reward_normalization=False,
            is_double=False,
            clip_loss_grad=False,  # if True, use huber loss
        )
        # Set Collectors
        self.collectors = {
            "Train": Collector(
                self.policy, self.envs["Train"], self.buffer, exploration_noise=True
            ),
            "Test": Collector(self.policy, self.envs["Test"], exploration_noise=True),
        }

    def createNN(self, model_name, model_cfg):
        NN = getattr(networks, model_name)(
            *self.state_shape,
            action_shape=self.action_shape,
            device=self.device,
            **model_cfg,
        )
        return NN.to(self.device)

    def run(self):
        """Run the game for multiple times"""
        self.logger.info("Start training ...")
        set_one_thread()
        start_time = time.time()
        # Load checkpoint
        if self.cfg["resume_from_log"]:
            self.load_checkpoint()
        set_random_seed(self.cfg["seed"])
        # Test train_collector and start filling replay buffer
        self.collectors["Train"].collect(
            n_step=self.batch_size * self.cfg["task"]["train_num"]
        )
        # Trainer
        result = OffpolicyTrainer(
            policy=self.policy,
            train_collector=self.collectors["Train"],
            test_collector=self.collectors["Test"],
            max_epoch=self.cfg["epoch"],
            step_per_epoch=self.cfg["step_per_epoch"],
            step_per_collect=self.cfg["step_per_collect"],
            episode_per_test=self.cfg["task"]["test_num"],
            batch_size=self.batch_size,
            update_per_step=self.cfg["update_per_step"],
            train_fn=self.train_fn,
            test_fn=self.test_fn,
            stop_fn=self.stop_fn,
            save_best_fn=self.save_model if self.cfg["save_model"] else None,
            logger=self.logger,
            verbose=True,
            # Set it to True to show speed, etc.
            show_progress=self.cfg["show_progress"],
            test_in_train=True,
            # Resume training setting
            resume_from_log=self.cfg["resume_from_log"],
            save_checkpoint_fn=self.save_checkpoint,
            # For gradient covariance computation
            compute_grad_covariance_at_epochs=self.cfg[
                "compute_grad_covariance_at_epochs"
            ],
            save_grad_covariance_dir=self.cfg["logs_dir"],
        ).run()
        for k, v in result.items():
            self.logger.info(f"{k}: {v}")

        # Save results
        self.save_result("Train")
        self.save_result("Test")
        end_time = time.time()
        self.logger.info(f"Memory usage: {rss_memory_usage():.2f} MB")
        self.logger.info(f"Time elapsed: {(end_time - start_time) / 60:.2f} minutes")

    def train_fn(self, epoch, env_step):
        # Linear decay epsilon in the first eps_steps
        if env_step <= self.cfg["agent"]["eps_steps"]:
            eps = self.cfg["agent"]["eps_start"] - env_step / self.cfg["agent"][
                "eps_steps"
            ] * (self.cfg["agent"]["eps_start"] - self.cfg["agent"]["eps_end"])
        else:
            eps = self.cfg["agent"]["eps_end"]
        self.policy.set_eps(eps)

    def test_fn(self, epoch, env_step):
        self.policy.set_eps(self.cfg["agent"]["eps_test"])

    def stop_fn(self, mean_rewards):
        if self.env.spec.reward_threshold:
            return mean_rewards >= self.env.spec.reward_threshold
        elif "Pong" in self.cfg["task"]["name"]:
            return mean_rewards >= 20
        else:
            return False

    def get_state_shape(self, env):
        if isinstance(env.observation_space, Discrete):
            return env.observation_space.n
        else:  # Box, MultiBinary
            return env.observation_space.shape

    def get_action_shape(self, env):
        if isinstance(env.action_space, Discrete):
            return env.action_space.n
        elif isinstance(env.action_space, Box):
            return env.action_space.shape
        else:
            raise ValueError("Unknown action type.")

    def save_model(self, model):
        torch.save(model.state_dict(), self.cfg["model_path"])

    def save_checkpoint(self, epoch, env_step, gradient_step):
        # Save model and optimizer states
        ckpt_dict = dict(
            model=self.policy.state_dict(), optim=self.policy.optim.state_dict()
        )
        torch.save(ckpt_dict, self.cfg["ckpt_path"])
        # Save results
        try:
            self.save_result("Train")
            self.save_result("Test")
        except:  # noqa: E722
            self.logger.info("Failed to save results")
        self.logger.info(f"Save checkpoint at epoch={epoch}")
        return self.cfg["ckpt_path"]

    def load_checkpoint(self):
        if os.path.exists(self.cfg["ckpt_path"]):
            ckpt_dict = torch.load(self.cfg["ckpt_path"], map_location=self.device)
            self.policy.load_state_dict(ckpt_dict["model"])
            self.policy.optim.load_state_dict(ckpt_dict["optim"])
            self.logger.info(
                f"Successfully restore policy and optim from: {self.cfg['ckpt_path']}."
            )
        else:
            self.logger.info(
                f"Checkpoint path: {self.cfg['ckpt_path']} does not exist."
            )

    def save_result(self, mode):
        # Convert tensorboard data to DataFrame, and save it.
        ea = EventAccumulator(self.cfg["logs_dir"])
        ea.Reload()
        # Get return
        tag = f"{mode.lower()}/reward"
        events = ea.Scalars(tag)
        result_list = []
        for event in events:
            result_dict = {
                "Task": self.task,
                "Model": self.model_name,
                "Step": event.step,
                "Perf": event.value,
            }
            result_list.append(result_dict)
        result_df = pd.DataFrame(result_list)
        result_df["Task"] = result_df["Task"].astype("category")
        result_df["Model"] = result_df["Model"].astype("category")
        result_df.to_feather(self.log_path[mode])

    def save_config(self):
        cfg_json = json.dumps(self.cfg, indent=2)
        f = open(self.cfg_path, "w")
        f.write(cfg_json)
        f.close()
