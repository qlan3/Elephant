import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import lax, random

import networks
from agents import SLTask
from utils.dataloader import load_data
from utils.helper import rss_memory_usage


class Classification(SLTask):
    """
    Classification task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.value_eps = 0.01
        self.percent_eps = 0.01
        try:
            self.output_dim = cfg["agent"]["model_cfg"]["output_dim"]
        except:  # noqa: E722
            self.output_dim = 10
        # Select loss function
        if self.cfg["srnn_lambda"] > 0:
            self.loss_fn = jax.jit(self.compute_SKL_loss)
        else:
            self.loss_fn = jax.jit(self.compute_loss)

    def createNN(self, model_name, model_cfg):
        NN = getattr(networks, model_name)(**model_cfg)
        return NN

    @partial(jax.jit, static_argnums=0)
    def compute_loss(self, params, state, batch):
        logits = state.apply_fn(params, batch["x"])["out"]
        one_hot = jax.nn.one_hot(batch["y"], self.output_dim)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        perf = jnp.mean(jnp.argmax(logits, -1) == batch["y"])
        return loss, perf

    @partial(jax.jit, static_argnums=0)
    def compute_SKL_loss(self, params, state, batch):
        """Compute Set KL loss + original loss"""
        outs = state.apply_fn(params, batch["x"])
        logits, feat = outs["out"], outs["feat"]
        # Compute orignal loss
        one_hot = jax.nn.one_hot(batch["y"], self.output_dim)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        perf = jnp.mean(jnp.argmax(logits, -1) == batch["y"])
        # Compute SKL loss
        feat = feat.mean(axis=0) + 1e-8
        skl_loss = jnp.where(
            lax.stop_gradient(feat > self.cfg["srnn_beta"]),
            self.cfg["srnn_beta"] / feat - jnp.log(self.cfg["srnn_beta"] / feat) - 1.0,
            0.0,
        )
        # Compute total loss
        loss += self.cfg["srnn_lambda"] * skl_loss.sum()
        return loss, perf

    @partial(jax.jit, static_argnums=0)
    def update_state(self, state, grads):
        return state.apply_gradients(grads=grads)

    def run(self):
        self.start_time = time.time()
        self.logger.info(f"Load dataset: {self.task}")
        self.seed, data_seed = random.split(self.seed)
        self.data = load_data(
            dataset=self.task, seed=data_seed, batch_size=self.cfg["batch_size"]
        )
        for mode in ["Train", "Test"]:
            self.logger.info(f"Datasize [{mode}]: {len(self.data[mode]['y'])}")
        self.logger.info("Create train state ...")
        self.logger.info("Create train state: build neural network model")
        model = self.createNN(self.model, self.cfg["agent"]["model_cfg"])
        self.seed, nn_seed = random.split(self.seed)
        params = model.init(nn_seed, self.data["dummy_input"])
        # Clone neural network params
        self.old_params = jax.tree_map(lambda x: x, params)
        self.old_fisher = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        self.logger.info("Create train state: set optimzer")
        optim = self.set_optim(self.cfg["optim"]["name"], self.cfg["optim"]["kwargs"])
        self.state = TrainState.create(apply_fn=model.apply, params=params, tx=optim)
        self.logger.info("Start training and evaluation ...")
        self.train_and_test()
        self.end_time = time.time()
        self.logger.info(f"Memory usage: {rss_memory_usage():.2f} MB")
        self.logger.info(
            f"Time elapsed: {(self.end_time - self.start_time) / 60:.2f} minutes"
        )

    def train_and_test(self):
        # Start training
        for epoch in range(1, self.cfg["epochs"] + 1):
            epoch_start_time = time.time()
            progress = epoch / self.cfg["epochs"]
            train_isnan = self.train_epoch(epoch, progress, self.cfg["batch_size"])
            test_isnan = self.evaluate_epoch(epoch, progress, self.cfg["batch_size"])
            if train_isnan or test_isnan:
                self.logger.info("NaN error detected!")
                break
            # Display speed
            if (epoch % self.cfg["display_interval"] == 0) or (progress == 1.0):
                speed = time.time() - epoch_start_time
                eta = (self.cfg["epochs"] - epoch) * speed / 60 if speed > 0 else -1
                self.logger.info(
                    f"<{self.config_idx}> {self.task} {self.model} Speed={speed:.2f} (s/epoch), ETA={eta:.2f} (mins)"
                )

    def train_epoch(self, epoch, progress, batch_size, mode="Train"):
        """Train for a single epoch."""
        self.seed, seed = random.split(self.seed)
        data_size = len(self.data[mode]["x"])
        batch_num = data_size // batch_size
        perms = random.permutation(seed, data_size)
        perms = perms[: batch_num * batch_size]  # Skip incomplete batch
        perms = perms.reshape((batch_num, batch_size))
        epoch_loss, epoch_perf = [], []
        for perm in perms:
            batch = {
                "x": self.data[mode]["x"][perm, ...],
                "y": self.data[mode]["y"][perm, ...],
            }
            # Forward: compute loss, performance, and gradient
            (loss, perf), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
                self.state.params, self.state, batch
            )
            # Backward: update train state
            self.state = self.update_state(self.state, grads)
            # Clone neural network params
            self.old_params = jax.tree_map(lambda x: x, self.state.params)
            # Log
            loss = float(jax.device_get(loss))
            perf = float(jax.device_get(perf))
            # Check NaN error
            if np.isnan(loss) or np.isnan(perf):
                return True
            epoch_loss.append(loss)
            epoch_perf.append(perf)
        epoch_loss = np.mean(epoch_loss)
        epoch_perf = np.mean(epoch_perf)
        # Save training result
        self.save_results(mode, epoch, progress, epoch_loss, epoch_perf)
        return False

    @partial(jax.jit, static_argnums=0)
    def evaluate_batch(self, state, batch):
        preds = state.apply_fn(state.params, batch["x"])
        logits, feat = preds["out"], preds["feat"]
        one_hot = jax.nn.one_hot(batch["y"], self.output_dim)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        perf = jnp.mean(jnp.argmax(logits, -1) == batch["y"])
        # Count number of active units
        is_dead = jnp.abs(lax.stop_gradient(feat)) < self.value_eps
        is_dead = jnp.sum(is_dead, axis=0)
        return loss, perf, is_dead

    def evaluate_epoch(self, epoch, progress, batch_size, mode="Test", key="Epoch"):
        assert mode in ["Test", "Valid"]
        batch_num = len(self.data[mode]["x"]) // batch_size
        data_size = batch_num * batch_size  # Skip incomplete batch
        perms = jnp.arange(data_size).reshape((batch_num, batch_size))
        epoch_loss, epoch_perf = [], []
        is_dead_sum = None
        for perm in perms:
            batch = {
                "x": self.data[mode]["x"][perm, ...],
                "y": self.data[mode]["y"][perm, ...],
            }
            # Forward
            loss, perf, is_dead = self.evaluate_batch(self.state, batch)
            # Log
            loss = float(jax.device_get(loss))
            perf = float(jax.device_get(perf))
            if is_dead_sum is None:
                is_dead_sum = is_dead
            else:
                is_dead_sum += is_dead
            # Check NaN error
            if np.isnan(loss) or np.isnan(perf):
                return True
            epoch_loss.append(loss)
            epoch_perf.append(perf)
        epoch_loss = np.mean(epoch_loss)
        epoch_perf = np.mean(epoch_perf)
        # Compute the number of active units
        active_unit_num = (is_dead_sum <= (1 - self.percent_eps) * data_size).sum()
        active_unit_num = int(jax.device_get(active_unit_num))
        # Save evaluation result
        self.save_results(
            mode,
            epoch,
            progress,
            epoch_loss,
            epoch_perf,
            key,
            active_unit_num=active_unit_num,
        )
        return False

    def save_results(
        self, mode, count, progress, loss, perf, key="Epoch", active_unit_num=-1
    ):
        """Save and display result."""
        result_dict = {
            "Task": self.task,
            "Model": self.model,
            key: count,
            "Loss": loss,
            "Perf": perf,
        }
        if active_unit_num >= 0:
            result_dict["Active"] = active_unit_num
        self.results[mode].append(result_dict)
        self.save_to_file(mode)
        if (count % self.cfg["display_interval"] == 0) or (progress == 1.0):
            self.logger.info(
                f"<{self.config_idx}> {self.task} {self.model} [{mode}] Progress {progress:.0%}, Loss={loss:.4f}, Perf={perf:.4f}"
            )
