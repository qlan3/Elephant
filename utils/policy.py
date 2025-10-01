import torch
import numpy as np
from typing import Optional

from tianshou.data import ReplayBuffer, to_torch_as
from tianshou.utils.net.discrete import sample_noise
from tianshou.policy import DQNPolicy as DQNPolicyBase
from tianshou.policy import RainbowPolicy as RainbowPolicyBase

from utils.helper import to_numpy


class DQNPolicy(DQNPolicyBase):
    def compute_grad_covariance(
        self,
        epoch: int,
        sample_size: int,
        buffer: Optional[ReplayBuffer],
        save_dir: str,
    ):
        assert buffer is not None, "Buffer is empty!"
        batches, indices = buffer.sample(sample_size)
        self.updating = True
        batches = self.process_fn(batches, buffer, indices)
        # Compute gradient per sample
        sample_grads = []
        for i in range(sample_size):
            batch = batches[None, i]  # prepend batch dimension for processing
            # Compute loss
            weight = batch.pop("weight", 1.0)
            q = self(batch).logits
            q = q[np.arange(len(q)), batch.act]
            returns = to_torch_as(batch.returns.flatten(), q)
            td_error = returns - q
            if self._clip_loss_grad:
                y = q.reshape(-1, 1)
                t = returns.reshape(-1, 1)
                loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
            else:
                loss = (td_error.pow(2) * weight).mean()
            # Gather gradient
            grad = torch.autograd.grad(loss, list(self.model.parameters()))
            grad = [g.reshape(-1) for g in grad]
            sample_grads.append(to_numpy(torch.cat(grad, -1)))
        self.updating = False
        # Save sample_grads to file
        sample_grads = np.array(sample_grads)
        np.save(save_dir + f"grad_{epoch}.npy", sample_grads)


class RainbowPolicy(RainbowPolicyBase):
    def compute_grad_covariance(
        self,
        epoch: int,
        sample_size: int,
        buffer: Optional[ReplayBuffer],
        save_dir: str,
    ):
        assert buffer is not None, "Buffer is empty!"
        batches, indices = buffer.sample(sample_size)
        self.updating = True
        batches = self.process_fn(batches, buffer, indices)
        # Compute gradient per sample
        sample_noise(self.model)
        if self._target and sample_noise(self.model_old):
            self.model_old.train()  # so that NoisyLinear takes effect
        sample_grads = []
        for i in range(sample_size):
            batch = batches[None, i]  # prepend batch dimension for processing
            # Compute loss
            with torch.no_grad():
                target_dist = self._target_dist(batch)
            weight = batch.pop("weight", 1.0)
            curr_dist = self(batch).logits
            act = batch.act
            curr_dist = curr_dist[np.arange(len(act)), act, :]
            cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(1)
            loss = (cross_entropy * weight).mean()
            # Gather gradient
            grad = torch.autograd.grad(loss, list(self.model.parameters()))
            grad = [g.reshape(-1) for g in grad]
            sample_grads.append(to_numpy(torch.cat(grad, -1)))
        self.updating = False
        # Save sample_grads to file
        sample_grads = np.array(sample_grads)
        np.save(save_dir + f"grad_{epoch}.npy", sample_grads)
