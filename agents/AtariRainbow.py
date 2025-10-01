from tianshou.data import Collector, PrioritizedVectorReplayBuffer

import networks
from agents.AtariDQN import AtariDQN
from utils.policy import RainbowPolicy


class AtariRainbow(AtariDQN):
    """
    Implementation of Rainbow for Atari games.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.buffer, self.policy, self.collectors
        # Set replay buffer: `save_last_obs` and `stack_num` can be removed when you have enough RAM
        self.buffer = PrioritizedVectorReplayBuffer(
            total_size=self.cfg["buffer_size"],
            buffer_num=self.cfg["task"]["train_num"],
            ignore_obs_next=True,
            save_only_last_obs=self.save_only_last_obs,
            stack_num=self.cfg["frames_stack"],
            alpha=self.cfg["agent"]["alpha"],
            beta=self.cfg["agent"]["beta_start"],
            weight_norm=True,
        )
        # Define policy
        self.policy = RainbowPolicy(
            model=self.net,
            optim=self.optimizer,
            discount_factor=self.discount,
            estimation_step=self.cfg["n_step"],
            target_update_freq=self.cfg["target_update_steps"],
            num_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            reward_normalization=False,
        ).to(self.device)
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
            num_atoms=51,
            noisy_std=0.1,
            device=self.device,
            **model_cfg,
        )
        self.logger.info(
            f"Number of parameters in net: {sum(p.numel() for p in NN.parameters())}"
        )
        return NN.to(self.device)

    def train_fn(self, epoch, env_step):
        # Linear decay epsilon in the first eps_steps
        if env_step <= self.cfg["agent"]["eps_steps"]:
            eps = self.cfg["agent"]["eps_start"] - env_step / self.cfg["agent"][
                "eps_steps"
            ] * (self.cfg["agent"]["eps_start"] - self.cfg["agent"]["eps_end"])
        else:
            eps = self.cfg["agent"]["eps_end"]
        self.policy.set_eps(eps)
        # Linear decay beta in the first beta_steps for priority replay
        if env_step <= self.cfg["agent"]["beta_steps"]:
            beta = self.cfg["agent"]["beta_start"] - env_step / self.cfg["agent"][
                "beta_steps"
            ] * (self.cfg["agent"]["beta_start"] - self.cfg["agent"]["beta_end"])
        else:
            beta = self.cfg["agent"]["beta_end"]
        self.buffer.set_beta(beta)
