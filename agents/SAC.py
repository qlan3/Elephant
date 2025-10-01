import copy
import time
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.RLTask import RLTask, TargetState, TrainState
from networks import MLPGaussianTanhActor, MLPQCritic, Temperature
from utils.replay import FiniteReplay


class SAC(RLTask):
    """
    Implementation of Soft Actor-Critic.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.action_type == "CONTINUOUS", (
            f"{self.model} only supports continous action spaces."
        )
        self.exploration_steps = max(
            int(self.cfg["agent"]["exploration_steps"]), self.cfg["batch_size"]
        )
        # Set replay buffer
        self.replay = FiniteReplay(
            self.cfg["buffer_size"],
            keys=["obs", "action", "reward", "mask", "next_obs"],
        )
        # Set networks and states
        self.createNN()
        self.states = {"critic": self.critic_state, "actor": self.actor_state}

    def createNN(self):
        # Create train_states and nets of actor, critic, and temperature
        dummy_obs = self.env["Train"].observation_space.sample()[None,]
        dummy_action = self.env["Train"].action_space.sample()[None,]
        self.seed, actor_seed, critic_seed, temp_seed = jax.random.split(self.seed, 4)
        self.actor_net = MLPGaussianTanhActor(
            action_size=self.action_size, **self.cfg["agent"]["model_cfg"]
        )
        self.critic_net = nn.vmap(
            MLPQCritic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},  # Parameters are not shared between critics
            split_rngs={"params": True},  # Different initializations
            axis_size=2,  # Number of critics
        )(**self.cfg["agent"]["model_cfg"])
        self.temp_net = Temperature(init_temp=1.0)
        actor_optim = self.set_optim(
            self.cfg["optim"]["name"], self.cfg["optim"]["kwargs"]
        )
        critic_optim = self.set_optim(
            self.cfg["optim"]["name"], self.cfg["optim"]["kwargs"]
        )
        actor_params = self.actor_net.init(actor_seed, dummy_obs)
        critic_params = self.critic_net.init(critic_seed, dummy_obs, dummy_action)
        # Set actor_state and critic_state
        self.actor_state = TrainState.create(
            apply_fn=self.actor_net.apply, params=actor_params, tx=actor_optim
        )
        self.critic_state = TargetState.create(
            apply_fn=self.critic_net.apply,
            params=critic_params,
            target_params=copy.deepcopy(critic_params),
            tx=critic_optim,
        )
        self.target_entropy = -0.5 * self.action_size
        self.temp_state = TrainState.create(
            apply_fn=self.temp_net.apply,
            params=self.temp_net.init(temp_seed),
            tx=self.set_optim(self.cfg["optim"]["name"], self.cfg["optim"]["kwargs"]),
        )
        num_params_actor = sum(
            p.size for p in jax.tree_flatten(self.actor_state.params)[0]
        )
        num_params_critic = sum(
            p.size for p in jax.tree_flatten(self.critic_state.params)[0]
        )
        self.logger.info(
            f"Number of parameters: {num_params_actor + num_params_critic}"
        )

    def run_steps(self, mode="Train"):
        self.start_time = time.time()
        obs, info = self.env[mode].reset()
        for step in range(self.train_steps):
            # Get an action
            action = self.get_action(step, obs[None,], mode)["action"]
            # Take a env step
            next_obs, reward, terminated, truncated, info = self.env[mode].step(action)
            # Save experience
            mask = self.discount * (1 - terminated)
            self.save_experience(obs, action, reward, mask, next_obs)
            # Update observation
            obs = next_obs
            # Record and reset
            if terminated or truncated:
                result_dict = {
                    "Task": self.task,
                    "Model": self.model,
                    "Step": step,
                    "Perf": info["episode"]["r"][0],
                }
                self.results[mode].append(result_dict)
                obs, info = self.env[mode].reset()
            # Update the model
            if (
                step > self.exploration_steps
                and step % self.cfg["agent"]["update_freq"] == 0
            ):
                batch = self.replay.sample(self.cfg["batch_size"])
                self.seed, critic_seed, action_seed = jax.random.split(self.seed, 3)
                self.critic_state = self.update_critic(
                    self.actor_state,
                    self.critic_state,
                    self.temp_state,
                    batch,
                    critic_seed,
                )
                self.actor_state, entropy = self.update_actor(
                    self.actor_state,
                    self.critic_state,
                    self.temp_state,
                    batch,
                    action_seed,
                )
                self.temp_state = self.update_temperature(self.temp_state, entropy)
            # Display log, test, and save checkpoint
            self.log_test_save(step, self.train_steps, mode)

    def get_action(self, step, obs, mode="Train"):
        if mode == "Train":
            if step <= self.exploration_steps:
                action = self.env[mode].action_space.sample()
            else:
                action, self.seed = self.random_action(
                    self.actor_state, self.critic_state, obs, self.seed
                )
                action = jax.device_get(action)[0]
        else:  # mode == 'Test'
            action, self.seed = self.optimal_action(
                self.actor_state, self.critic_state, obs, self.seed
            )
            action = jax.device_get(action)[0]
        return dict(action=action)

    @partial(jax.jit, static_argnames=["self"])
    def random_action(self, actor_state, critic_state, obs, seed):
        seed, action_seed = jax.random.split(seed, 2)
        u_mean, u_log_std = actor_state.apply_fn(actor_state.params, obs)
        eps = jax.random.normal(action_seed, shape=u_mean.shape)
        u = u_mean + jnp.exp(u_log_std) * eps
        action = jnp.tanh(u)
        return action, seed

    @partial(jax.jit, static_argnames=["self"])
    def optimal_action(self, actor_state, critic_state, obs, seed):
        u_mean, _ = actor_state.apply_fn(actor_state.params, obs)
        action = jnp.tanh(u_mean)
        return action, seed

    @partial(jax.jit, static_argnames=["self"])
    def update_critic(self, actor_state, critic_state, temp_state, batch, seed):
        next_action, next_logp = self.sample_action_with_logp(
            actor_state, actor_state.params, batch["next_obs"], seed
        )
        q_next = critic_state.apply_fn(
            critic_state.target_params, batch["next_obs"], next_action
        )  # Shape: (critic, batch, 1)
        q_next = jnp.min(q_next, axis=0)  # Shape: (batch, 1)
        alpha = temp_state.apply_fn(temp_state.params)
        q_target = batch["reward"].reshape(-1, 1) + batch["mask"].reshape(-1, 1) * (
            q_next - alpha * next_logp.reshape(-1, 1)
        )  # Shape: (batch, 1)

        # Compute critic loss
        def critic_loss(params):
            qs = critic_state.apply_fn(
                params, batch["obs"], batch["action"]
            )  # Shape: (critic, batch, 1)
            loss = ((qs - q_target) ** 2).mean(axis=1).sum()
            return loss

        grads = jax.grad(critic_loss)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=grads)
        # Soft-update target network
        critic_state = critic_state.replace(
            target_params=optax.incremental_update(
                critic_state.params,
                critic_state.target_params,
                self.cfg["agent"]["tau"],
            )
        )
        return critic_state

    @partial(jax.jit, static_argnames=["self"])
    def update_actor(self, actor_state, critic_state, temp_state, batch, seed):
        alpha = temp_state.apply_fn(temp_state.params)

        # Compute actor loss
        def actor_loss(params):
            action, logp = self.sample_action_with_logp(
                actor_state, params, batch["obs"], seed
            )
            entropy = -logp.mean()
            qs = critic_state.apply_fn(critic_state.params, batch["obs"], action)
            q_min = jnp.min(qs, axis=0)  # Shape: (batch, 1)
            loss = (alpha * logp - q_min).mean()
            return loss, entropy

        grads, entropy = jax.grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, entropy

    @partial(jax.jit, static_argnames=["self"])
    def update_temperature(self, temp_state, entropy):
        def temperature_loss(params):
            alpha = temp_state.apply_fn(params)
            loss = alpha * (entropy - self.target_entropy).mean()
            return loss

        grads = jax.grad(temperature_loss)(temp_state.params)
        temp_state = temp_state.apply_gradients(grads=grads)
        return temp_state

    @partial(jax.jit, static_argnames=["self"])
    def sample_action_with_logp(self, actor_state, params, obs, seed):
        u_mean, u_log_std = actor_state.apply_fn(params, obs)
        eps = jax.random.normal(seed, shape=u_mean.shape)
        u = u_mean + jnp.exp(u_log_std) * eps
        action = jnp.tanh(u)
        # Get log_prob(action): https://github.com/openai/spinningup/issues/279
        logp = (-0.5 * (eps**2) - 0.5 * jnp.log(2.0 * jnp.pi) - u_log_std).sum(axis=-1)
        logp -= (2 * (jnp.log(2) - u - nn.softplus(-2 * u))).sum(axis=-1)
        return action, logp
