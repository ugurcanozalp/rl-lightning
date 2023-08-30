
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

from ..nets import policy_value_map


class OQLMemory(Memory):

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
            action, log_prob, value = agent.step(self.observation)
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


class OQL(pl.LightningModule):
    
    def __init__(self, 
        env: str = "HalfCheetah-v4", 
        piv_net: str = "continuous_mlp2", 
        gamma: float = 0.99,
        alpha: float = 0.2,
        buffer_capacity: int = 1000000,
        tau: float = 0.005,
        start_steps: int = 10000,
        steps_per_epoch: int = 512,
        batch_per_step: int = 1,
        lr: float = 3e-4,
        batch_size: int = 256, 
        **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self._epoch_size = self.hparams.batch_per_step * self.hparams.batch_size * self.hparams.steps_per_epoch
        self._memory = OQLMemory(
            gamma=self.hparams.gamma, 
            env=gym.make(env, render_mode="human"),
            capacity=self.hparams.buffer_capacity, 
            epoch_size=self._epoch_size)
        self._piv = policy_value_map[piv_net](**self.env_info)
        self._piv_target = policy_value_map[piv_net](**self.env_info)
        for param in self._piv_target.parameters():
            param.requires_grad = False
        self._hard_update(self._piv, self._piv_target)
        self._memory.reset()
        self._memory.rollout(self, self.hparams.start_steps) # initial rollout to fill buffer!
        self.total_env_interactions = 0

    @property
    def env_info(self):
        return {
            "observation_shape": self._memory._env.observation_space.shape,
            "action_shape": self._memory._env.action_space.shape
        }

    def forward(self, observation: th.Tensor):
        distr, value = self._piv(observation)
        action = distr.rsample().clip(-1+1e-5, 1-1e-5)
        log_prob = distr.log_prob(action)
        if self._piv.independent_actions:
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob, value

    @th.no_grad()
    def step(self, observation: np.ndarray):
        observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
        action_, log_prob_, value_ = self.forward(observation_)
        action = action_.squeeze(0).cpu().numpy()
        log_prob = log_prob_.squeeze(0).cpu().numpy()
        value = value_.squeeze(0).cpu().numpy()
        return action, log_prob, value

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
        piv_optim = self.optimizers()
        piv_optim.zero_grad()
        action_distr, value = self._piv(observation)
        information_action = - action_distr.log_prob(action)
        # entropy_policy = - action_distr.log_prob(action_distr.sample((10, ))).mean(dim=0).detach()
        if self._piv.independent_actions:
            information_action = information_action.sum(dim=-1)
            # entropy_policy = entropy_policy.sum(dim=-1)
        with th.no_grad():
            _, next_value = self._piv_target(next_observation) 
        value_rhs = self.hparams.alpha * information_action + reward + self.hparams.gamma * next_value * done.logical_not()
        policy_value_loss = th.nn.functional.mse_loss(value, value_rhs) 
        self.manual_backward(policy_value_loss)
        piv_optim.step()
        self._soft_update(self._piv, self._piv_target)
        info = {
            "policy_value_loss": policy_value_loss, 
            "policy_cross_entropy": information_action.mean()
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
        piv_optim = Adam(self._piv.parameters(), lr=self.hparams.lr)
        return piv_optim

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset = self._memory,
            batch_size = self.hparams.batch_size, 
            pin_memory=True, 
        )
        return dataloader

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--env", type=str, default="HalfCheetah-v4")
        parser.add_argument("--piv_net", type=str, default="continuous_mlp2")
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--buffer_capacity", type=int, default=1000000)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--start_steps", type=int, default=10000)
        parser.add_argument("--steps_per_epoch", type=int, default=512)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--target_update_interval", type=int, default=1)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass