import time

import flax
import numpy as np
from flax.training.train_state import TrainState
from gymnasium import spaces

from agents import SLTask
from utils.env import make_env
from utils.helper import rss_memory_usage


class TargetState(TrainState):
    target_params: flax.core.FrozenDict = None


class RLTask(SLTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg["task"].setdefault("kwargs", {})
        self.cfg["agent"].setdefault("update_freq", 1)
        self.discount = self.cfg["discount"]
        self.batch_size = cfg["batch_size"]
        self.train_steps = int(cfg["train_steps"])
        self.cfg["test_interval"] = int(cfg["test_interval"])
        self.cfg["display_interval"] = int(cfg["display_interval"])
        # Make environment
        self.env = {
            "Train": make_env(self.task, **self.cfg["task"]["kwargs"]),
            "Test": make_env(
                self.task,
                deque_size=self.cfg["test_episodes"],
                **self.cfg["task"]["kwargs"],
            ),
        }
        self.env["Train"].set_seed(self.cfg["seed"])
        self.env["Test"].set_seed(self.cfg["seed"] + 42)
        self.obs_size = self.get_obs_size(self.env["Train"])
        self.action_type, self.action_size = self.get_action_size(self.env["Train"])

    def createNN(self, model_name, model_cfg):
        raise NotImplementedError

    def run(self):
        self.logger.info("Start training ...")
        start_time = time.time()
        self.run_steps()
        end_time = time.time()
        self.logger.info(f"Memory usage: {rss_memory_usage():.2f} MB")
        self.logger.info(f"Time elapsed: {(end_time - start_time) / 60:.2f} minutes")

    def run_steps(self, mode="Train"):
        raise NotImplementedError

    def save_experience(self, obs, action, reward, mask, next_obs):
        prediction = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "mask": mask,
            "next_obs": next_obs,
        }
        self.replay.add(prediction)

    def get_action_size(self, env):
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "single_action_space"):
            action_space = env.unwrapped.single_action_space
        else:
            action_space = env.action_space
        if isinstance(action_space, spaces.Discrete):
            action_type = "DISCRETE"
            return action_type, action_space.n
        elif isinstance(action_space, spaces.Box):
            action_type = "CONTINUOUS"
            return action_type, int(np.prod(action_space.shape))
        else:
            action_type = "UNKNOWN"
            raise ValueError("Unknown action type.")

    def get_obs_size(self, env):
        if hasattr(env, "unwrapped") and hasattr(
            env.unwrapped, "single_observation_space"
        ):
            observation_space = env.unwrapped.single_observation_space
        else:
            observation_space = env.observation_space
        if isinstance(observation_space, spaces.Discrete):
            return observation_space.n
        elif isinstance(observation_space, spaces.Box):
            return int(np.prod(observation_space.shape))
        else:
            raise ValueError("Unknown observation type.")

    def log_test_save(self, iter, train_iter, mode):
        if self.cfg["agent"]["name"] in ["PPO"]:
            step = iter * self.cfg["agent"]["collect_steps"]
        else:
            step = iter
        # Test for several episodes
        if (self.cfg["test_interval"] > 0) and (
            iter % self.cfg["test_interval"] == 0 or iter == train_iter - 1
        ):
            self.test(step)
        # Save to file
        if (self.cfg["ckpt_interval"] > 0) and (
            (iter > 0 and iter % self.cfg["ckpt_interval"] == 0)
            or iter == train_iter - 1
        ):
            if self.cfg["test_interval"] > 0:
                modes = ["Train", "Test"]
            else:
                modes = ["Train"]
            for mode in modes:
                self.save_to_file(mode)
        # Display log
        if (self.cfg["display_interval"] > 0) and (
            iter % self.cfg["display_interval"] == 0 or iter == train_iter - 1
        ):
            speed = step / (time.time() - self.start_time)
            eta = (self.train_steps - step) / speed / 60 if speed > 0 else -1
            self.logger.info(
                f"<{self.config_idx}> ({self.model}) ({self.task}) {step}/{self.train_steps}: Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)"
            )
            if len(self.results[mode]) > 0:
                self.logger.info(
                    f"<{self.config_idx}> [{mode}] {step}/{self.train_steps}: Return={self.results[mode][-1]['Perf']:.2f}"
                )

    def test(self, step, mode="Test"):
        for _ in range(self.cfg["test_episodes"]):
            obs, info = self.env[mode].reset()
            while True:
                action = self.get_action(step, obs[None,], mode)["action"]
                obs, reward, terminated, truncated, info = self.env[mode].step(action)
                if terminated or truncated:
                    break
        # Gather result
        result_dict = {
            "Task": self.task,
            "Model": self.model,
            "Step": step,
            "Perf": np.mean(self.env[mode].return_queue),
        }
        self.results[mode].append(result_dict)
        self.logger.info(
            f"<{self.config_idx}> [{mode}] Step {step}/{self.train_steps}: Return={result_dict['Return']:.2f}"
        )
