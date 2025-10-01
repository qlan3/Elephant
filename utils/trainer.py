import tqdm
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import BaseTrainer, OffpolicyTrainer
from tianshou.trainer.utils import gather_info
from tianshou.utils import (
    BaseLogger,
    DummyTqdm,
    LazyLogger,
    tqdm_config,
)


class OffpolicyTrainerGradCovariance(OffpolicyTrainer):
    """
    Add gradient covariance computation to OffpolicyTrainer.
    """

    __doc__ = BaseTrainer.gen_doc("offpolicy") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Optional[Collector],
        max_epoch: int,
        step_per_epoch: int,
        step_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        update_per_step: Union[int, float] = 1,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        compute_grad_covariance_at_epochs: List[int] = None,
        save_grad_covariance_dir: str = None,
        **kwargs: Any,
    ):
        super().__init__(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            update_per_step=update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            test_in_train=test_in_train,
            **kwargs,
        )
        self.compute_grad_covariance_at_epochs = compute_grad_covariance_at_epochs
        self.save_grad_covariance_dir = save_grad_covariance_dir

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:
            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration
            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        # set policy in train mode
        self.policy.train()

        # Compute gradient covariance matrix
        if (
            self.compute_grad_covariance_at_epochs is not None
            and self.epoch in self.compute_grad_covariance_at_epochs
        ):
            self.policy.compute_grad_covariance(
                self.epoch,
                self.batch_size,
                self.train_collector.buffer,
                self.save_grad_covariance_dir,
            )

        epoch_stat: Dict[str, Any] = dict()

        if self.show_progress:
            progress = tqdm.tqdm
        else:
            progress = DummyTqdm

        # perform n step_per_epoch
        with progress(
            total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total and not self.stop_fn_flag:
                data: Dict[str, Any] = dict()
                result: Dict[str, Any] = dict()
                if self.train_collector is not None:
                    data, result, self.stop_fn_flag = self.train_step()
                    t.update(result["n/st"])
                    if self.stop_fn_flag:
                        t.set_postfix(**data)
                        break
                else:
                    assert self.buffer, "No train_collector or buffer specified"
                    result["n/ep"] = len(self.buffer)
                    result["n/st"] = int(self.gradient_step)
                    t.update()

                self.policy_update_fn(data, result)
                t.set_postfix(**data)

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        # for offline RL
        if self.train_collector is None:
            self.env_step = self.gradient_step * self.batch_size

        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch, self.env_step, self.gradient_step, self.save_checkpoint_fn
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()
                if not self.is_run:
                    epoch_stat.update(test_stat)

        if not self.is_run:
            epoch_stat.update({k: v.get() for k, v in self.stat.items()})
            epoch_stat["gradient_step"] = self.gradient_step
            epoch_stat.update(
                {
                    "env_step": self.env_step,
                    "rew": self.last_rew,
                    "len": int(self.last_len),
                    "n/ep": int(result["n/ep"]),
                    "n/st": int(result["n/st"]),
                }
            )
            info = gather_info(
                self.start_time,
                self.train_collector,
                self.test_collector,
                self.best_reward,
                self.best_reward_std,
            )
            return self.epoch, epoch_stat, info
        else:
            return None
