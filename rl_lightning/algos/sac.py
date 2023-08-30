
import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser

import gym
import numpy as np
import torch as th
import  pytorch_lightning as pl
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from ..memory import Memory

from ..nets import policy_map, qvalue_map


class SACMemory(Memory):

    insert_fields = ("observation", "action", "reward", "next_observation", "done", "truncated")
    derived_fields = ()

    def __init__(self, gamma: float, env: gym.Env, **kwargs):
        self._gamma = gamma
        self._env = env
        self.observation, _ = self._env.reset()
        Memory.__init__(self, **kwargs)

    def reset(self):
        self.last_episode_score = 0
        self.episode_score = 0
        self.clear()
        self.observation, _ = self._env.reset()

    def rollout(self, agent: Callable, num_steps: int):
        for t in range(num_steps):
            action = agent.step(self.observation)
            next_observation, reward, done, truncated, _ = self._env.step(action)
            self.episode_score += reward
            self.step(
                np.float32(self.observation), 
                np.float32(action), 
                np.float32(reward),
                np.float32(next_observation), 
                done,
                truncated,
            )
            if done or truncated:
                self.observation, _ = self._env.reset()
                agent.reset()
                self.last_episode_score = self.episode_score
                self.episode_score = 0
            else:
                self.observation = next_observation
        return self.last_episode_score

    def compute_function(self,
            agent, 
            observation, 
            action, 
            reward, 
            next_observation, 
            done, 
            truncated):
        return ()


class SAC(pl.LightningModule):
    
    def __init__(self, 
        env: str = "HalfCheetah-v4", 
        pi_net: str = "continuous_mlp2", 
        q_net: str = "continuous_mlp2",
        gamma: float = 0.99,
        alpha: float = 0.2, 
        buffer_capacity: int = 1000000,
        tau: float = 0.005, 
        start_steps: int = 10000, 
        steps_per_epoch: int = 512, 
        batch_per_step: int = 1, 
        pi_lr: float = 3e-4,
        q_lr: float = 3e-4, 
        batch_size: int = 256, 
        **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self._epoch_size = self.hparams.batch_per_step * self.hparams.batch_size * self.hparams.steps_per_epoch
        self._memory = SACMemory(
            gamma=self.hparams.gamma, 
            env=gym.make(env, render_mode="human"),
            capacity=self.hparams.buffer_capacity, 
            epoch_size=self._epoch_size)
        self._pi = policy_map[pi_net](**self.env_info)
        self._q1 = qvalue_map[q_net](**self.env_info)
        self._q2 = qvalue_map[q_net](**self.env_info)
        self._q1_target = qvalue_map[q_net](**self.env_info).eval()
        self._q2_target = qvalue_map[q_net](**self.env_info).eval()
        self._hard_update(self._q1, self._q1_target)
        self._hard_update(self._q2, self._q2_target)
        self._memory.reset()
        self._memory.rollout(self, self.hparams.start_steps) # initial rollout to fill buffer!
        self.total_env_interactions = 0

    @property
    def env_info(self):
        return {
            "observation_shape": self._memory._env.observation_space.shape,
            "action_shape": self._memory._env.action_space.shape, 
            "observation_space": self._memory._env.observation_space, 
            "action_space": self._memory._env.action_space, 
        }

    def forward(self, observation: th.Tensor):
        distr = self._pi(observation)
        action = distr.rsample()
        return action

    @th.no_grad()
    def step(self, observation: np.ndarray):
        observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
        action_ = self.forward(observation_)
        action = action_.squeeze(0).cpu().numpy()
        return action

    def reset(self):
        pass

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.hparams.tau*local_param.data + (1.0-self.hparams.tau)*target_param.data)

    def _hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
            
    def training_step(self, batch, batch_index):
        observation, action, reward, next_observation, done, truncated = batch
        pi_optim, q_optim = self.optimizers()
        with th.no_grad():
            next_action_distr = self._pi(next_observation)
            next_action = next_action_distr.sample()
            next_entropy = - next_action_distr.log_prob(next_action)
            if self._pi.independent_actions: 
                next_entropy = next_entropy.sum(dim=-1)
            next_qvalue1 = self._q1_target(next_observation, next_action)
            next_qvalue2 = self._q2_target(next_observation, next_action)
            next_qvalue_target = th.min(next_qvalue1, next_qvalue2) + self.hparams.alpha * next_entropy
            qvalue = reward + (self.hparams.gamma * next_qvalue_target * done.logical_not())
        # update critics
        q_optim.zero_grad()
        qvalue1_est = self._q1(observation, action)
        qvalue1_loss = th.nn.functional.mse_loss(qvalue1_est, qvalue)
        qvalue2_est = self._q2(observation, action)
        qvalue2_loss = th.nn.functional.mse_loss(qvalue2_est, qvalue)
        qvalue_loss = 0.5*qvalue1_loss + 0.5*qvalue2_loss
        self.manual_backward(qvalue_loss)
        q_optim.step()
        # update actor
        pi_optim.zero_grad()
        action_distr = self._pi(observation)
        action_imaginary = action_distr.rsample()
        entropy = - action_distr.log_prob(action_imaginary)
        if self._pi.independent_actions: 
            entropy = entropy.sum(dim=-1)
        q_imaginary = th.min(self._q1(observation, action_imaginary), self._q2(observation, action_imaginary))
        pi_loss = -(q_imaginary + self.hparams.alpha * entropy).mean()
        self.manual_backward(pi_loss)
        pi_optim.step()
        self._soft_update(self._q1, self._q1_target)
        self._soft_update(self._q2, self._q2_target)
        info = {
            "pi_loss": pi_loss,
            "q1_loss": qvalue1_loss, 
            "q2_loss": qvalue2_loss,
            "entropy": entropy.mean(),
        }
        for key, value in info.items():
            self.log(key, value)
        return info
        
    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        observation, action, reward, next_observation, done, truncated = batch
        return {}

    def on_train_start(self):
        self.total_env_interactions = 0

    def on_train_epoch_start(self):
        last_episode_score = self._memory.rollout(self, self.hparams.steps_per_epoch) 
        self.total_env_interactions += self.hparams.steps_per_epoch
        self._memory.compute(self) # compute the necessary things!
        self.log("last_episode_score", last_episode_score)
        self.log("total_env_interactions", self.total_env_interactions)

    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        pi_optim = Adam(self._pi.parameters(), lr=self.hparams.pi_lr)
        # q_optim = Adam(self._q1.parameters(), lr=self.hparams.q_lr)
        q_optim = Adam(
            [{'params': self._q1.parameters()}, {'params': self._q2.parameters()}], 
            lr=self.hparams.q_lr
        )
        return [pi_optim, q_optim]

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset = self._memory,
            batch_size = self.hparams.batch_size, 
            pin_memory=True, 
        )
        return dataloader

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--env", type=str, default="HalfCheetah-v4")
        parser.add_argument("--pi_net", type=str, default="continuous_mlp2")
        parser.add_argument("--q_net", type=str, default="continuous_mlp2")
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--buffer_capacity", type=int, default=1000000)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--start_steps", type=int, default=10000)
        parser.add_argument("--steps_per_epoch", type=int, default=512)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--target_update_interval", type=int, default=1)
        parser.add_argument("--pi_lr", type=float, default=3e-4)
        parser.add_argument("--q_lr", type=float, default=3e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass