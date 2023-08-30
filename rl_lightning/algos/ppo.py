
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

from ..nets import policy_map, value_map


class PPOMemory(Memory):

    insert_fields = ("observation", "action", "reward", "next_observation", "done", "truncated", "log_prob", "value")
    derived_fields = ("cum_return", "gae")

    def __init__(self, gamma: float, lambd: float, env: gym.Env, **kwargs):
        self._gamma = gamma
        self._lambd = lambd
        self._env = env
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
                np.float32(log_prob),
                np.float32(value)
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
            truncated, 
            log_prob, 
            value):
        not_done = np.logical_not(done)
        cum_return, gae = np.zeros_like(reward), np.zeros_like(reward)
        _, _, last_value = agent.step(next_observation[-1])
        cum_return[-1] = reward[-1] + not_done[-1] * self._gamma * last_value 
        delta = reward[-1] + self._gamma * last_value - value[-1] # delta at last time...
        gae[-1] = delta
        for t in reversed(range(self._not_computed-1)):
            if truncated[t]:
                cum_return[t] = value[t]
            else:
                cum_return[t] = reward[t] + not_done[t] * self._gamma * cum_return[t+1] 
            delta = reward[t] + not_done[t] * self._gamma * value[t+1] - value[t] # one step td error
            gae[t] = delta + not_done[t] * self._gamma * self._lambd * gae[t+1]
        return cum_return, gae

# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
class PPO(pl.LightningModule):
    
    def __init__(self, 
        env: str = "HalfCheetah-v4", 
        pi_net: str = "continuous_mlp2", 
        v_net: str = "continuous_mlp2",
        gamma: float = 0.99,
        lambd: float = 0.90, 
        alpha: float = 5e-4, 
        steps_per_rollout: int = 512, 
        subepochs_per_rollout: int = 20, 
        clip_ratio: int = 0.4, 
        pi_lr: float = 3e-5,
        v_lr: float = 3e-5, 
        vf_coef: float = 0.5, 
        max_grad_norm: float = 0.5, 
        batch_size: int = 128, 
        **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self._memory = PPOMemory(
            gamma=self.hparams.gamma, 
            lambd=self.hparams.lambd, 
            env=gym.make(env, render_mode="human"),
            capacity=self.hparams.steps_per_rollout, 
            epoch_size=self.hparams.steps_per_rollout * self.hparams.subepochs_per_rollout)
        self._pi = policy_map[pi_net](**self.env_info)
        self._v = value_map[v_net](**self.env_info)
        self._memory.reset()
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
        log_prob = distr.log_prob(action)
        if self._pi.independent_actions: 
            log_prob = log_prob.sum(dim=-1)
        value = self._v(observation)
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

    def training_step(self, batch, batch_index):
        observation, action, reward, next_observation, \
            done, truncated, log_prob, value, \
            cum_return, gae  = batch
        distr = self._pi(observation)
        cross_log_prob = distr.log_prob(action)
        entropy = - distr.log_prob(distr.rsample())
        if self._pi.independent_actions: 
            cross_log_prob = cross_log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        value = self._v(observation)
        #
        pi_optim, v_optim = self.optimizers()
        # policy loss        
        ratio = th.exp(cross_log_prob - log_prob)
        gae_normalized = (gae - gae.mean() ) / (gae.std() + 1e-6)
        clip_adv = th.clamp(ratio, 1 - self.hparams.clip_ratio, 1 + self.hparams.clip_ratio) * gae_normalized
        pi_loss = -(th.min(ratio * gae_normalized, clip_adv)).mean() - self.hparams.alpha * entropy.mean() 
        pi_optim.zero_grad()
        self.manual_backward(pi_loss)
        self.clip_gradients(pi_optim, grq_critic_mapadient_clip_val=self.hparams.max_grad_norm, gradient_clip_algorithm="norm")
        pi_optim.step()
        # critic loss
        v_loss = self.hparams.vf_coef * th.nn.functional.mse_loss(value, cum_return)
        v_optim.zero_grad()
        self.manual_backward(v_loss)
        self.clip_gradients(v_optim, gradient_clip_val=self.hparams.max_grad_norm, gradient_clip_algorithm="norm")
        v_optim.step() 
        # useful extra info
        approx_kl = (log_prob - cross_log_prob).mean().item()
        clipped = th.logical_or(ratio.gt(1 + self.hparams.clip_ratio), ratio.lt(1 - self.hparams.clip_ratio))
        clipfrac = th.as_tensor(clipped, dtype=th.float32).mean().item()
        #
        info = {
            "pi_loss": pi_loss,
            "v_loss": v_loss, 
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "cross_log_prob": cross_log_prob.mean(),
        }
        for key, value in info.items():
            self.log(key, value)
        return info
        
    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        observation, action, reward, next_observation, \
            done, truncated, log_prob, value, cum_return, gae = batch
        return {}

    def on_train_start(self):
        self.total_env_interactions = 0

    def on_train_epoch_start(self):
        # if self.current_epoch % self.hparams.subepochs_per_rollout == 0:
        last_episode_score = self._memory.rollout(self, self.hparams.steps_per_rollout) 
        self.total_env_interactions += self.hparams.steps_per_rollout
        self._memory.compute(self) # compute the necessary things!
        self.log("last_episode_score", last_episode_score)
        self.log("total_env_interactions", self.total_env_interactions)

    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        pi_optim = Adam(self._pi.parameters(), lr=self.hparams.pi_lr)
        v_optim = Adam(self._v.parameters(), lr=self.hparams.v_lr)
        return [pi_optim, v_optim]

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
        parser.add_argument("--pi_net", type=str, default="continuous_mlp2") 
        parser.add_argument("--v_net", type=str, default="continuous_mlp2")
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--lambd", type=float, default=0.90) 
        parser.add_argument("--alpha", type=float, default=5e-4)
        parser.add_argument("--steps_per_rollout", type=int, default=512) 
        parser.add_argument("--subepochs_per_rollout", type=int, default=20) 
        parser.add_argument("--clip_ratio", type=int, default=0.4) 
        parser.add_argument("--pi_lr", type=float, default=3e-5)
        parser.add_argument("--v_lr", type=float, default=3e-5)
        parser.add_argument("--vf_coef", type=float, default=0.5)
        parser.add_argument("--max_grad_norm", type=float, default=0.5)
        parser.add_argument("--batch_size", type=int, default=128)
        return parser

if __name__=="__main__":
    pass