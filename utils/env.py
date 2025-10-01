import gymnasium as gym  # noqa: F401
import gym_pygame  # noqa: F401
import numpy as np
import warnings
import envpool
from gymnasium import spaces
from gymnasium.wrappers import (
    ClipAction,
    FlattenObservation,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    RescaleAction,
    TransformObservation,
    TransformReward,
)


class UniversalSeed(gym.Wrapper):
    def set_seed(self, seed: int):
        _, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)


def make_env(env_name, deque_size=1, **kwargs):
    """
    Make env for general tasks.
    """
    env = gym.make(env_name, **kwargs)
    # Episode statistics wrapper
    env = RecordEpisodeStatistics(env, deque_size=deque_size)
    # Action wrapper
    if isinstance(env.action_space, spaces.Box):  # Continuous action space
        env = ClipAction(RescaleAction(env, min_action=-1, max_action=1))
    # Seed wrapper: must be the last wrapper to be effective
    env = UniversalSeed(env)
    return env


def ppo_make_env(env_name, gamma=0.99, deque_size=1, **kwargs):
    """Make env for PPO."""
    env = gym.make(env_name, **kwargs)
    # Episode statistics wrapper: set it before reward wrappers
    env = RecordEpisodeStatistics(env, deque_size=deque_size)
    # Action wrapper
    env = ClipAction(RescaleAction(env, min_action=-1, max_action=1))
    # Obs wrapper
    env = FlattenObservation(env)  # For dm_control
    env = NormalizeObservation(env)
    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # Reward wrapper
    env = NormalizeReward(env, gamma=gamma)
    env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    # Seed wrapper: must be the last wrapper to be effective
    env = UniversalSeed(env)
    return env


def sac_make_env(env_name, deque_size=1, **kwargs):
    """Make env for PPO."""
    env = gym.make(env_name, **kwargs)
    # Episode statistics wrapper: set it before reward wrappers
    env = RecordEpisodeStatistics(env, deque_size=deque_size)
    # Action wrapper
    env = ClipAction(RescaleAction(env, min_action=-1, max_action=1))
    # Obs wrapper
    env = FlattenObservation(env)  # For dm_control
    env = NormalizeObservation(env)
    # Seed wrapper: must be the last wrapper to be effective
    env = UniversalSeed(env)
    return env


def make_vec_env(
    env_name,
    num_envs=1,
    asynchronous=False,
    deque_size=1,
    max_episode_steps=None,
    **kwargs,
):
    env = gym.make(env_name, **kwargs)
    wrappers = []
    # Episode statistics wrapper
    wrappers.append(lambda env: RecordEpisodeStatistics(env, deque_size=deque_size))
    # Action wrapper
    if isinstance(env.action_space, spaces.Box):  # Continuous action space
        wrappers.append(
            lambda env: ClipAction(RescaleAction(env, min_action=-1, max_action=1))
        )
    envs = gym.vector.make(
        env_name,
        num_envs=num_envs,
        asynchronous=asynchronous,
        max_episode_steps=max_episode_steps,
        wrappers=wrappers,
    )
    # Seed wrapper: must be the last wrapper to be effective
    envs = UniversalSeed(envs)
    return envs


# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
def make_atari_env(task, seed, training_num, test_num, **kwargs):
    """Wrapper function for Atari env.
    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.
    :return: a tuple of (single env, training envs, test envs).
    """
    assert envpool is not None, "Please install envpool"
    if kwargs.get("scale", 0):
        warnings.warn(
            "EnvPool does not include ScaledFloatFrame wrapper, "
            "please set `x = x / 255.0` inside CNN network's forward function."
        )
    # parameters convertion
    train_envs = env = envpool.make_gym(
        task.replace("NoFrameskip-v4", "-v5"),
        num_envs=training_num,
        seed=seed,
        episodic_life=True,
        reward_clip=True,
        stack_num=kwargs.get("frame_stack", 4),
    )
    test_envs = envpool.make_gym(
        task.replace("NoFrameskip-v4", "-v5"),
        num_envs=test_num,
        seed=seed,
        episodic_life=False,
        reward_clip=False,
        stack_num=kwargs.get("frame_stack", 4),
    )
    return env, train_envs, test_envs
