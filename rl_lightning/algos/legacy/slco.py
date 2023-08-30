
import os
import math
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser

import gym
import numpy as np
import torch as th
import  pytorch_lightning as pl
from torch.optim import Adam, AdamW, Optimizer
from torch.utils.data import DataLoader

from ..memory import Memory

from ..nets import policy_map, value_map, model_map, reward_map


def eval_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    return total_norm

class SLCOMemory(Memory):

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


class SLCO(pl.LightningModule):
    
    def __init__(self, 
        env: str = "HalfCheetah-v4", 
        policy_net: str = "continuous_mlp2", 
        value_net: str = "continuous_mlp2", 
        model_net: str = "continuous_mlp2", 
        reward_net: str = "continuous_mlp2", 
        gamma: float = 0.99,
        alpha: float = 0.2, 
        beta: float = 0.05, 
        buffer_capacity: int = 1000000, 
        tau: float = 0.005, 
        start_steps: int = 10000, 
        steps_per_epoch: int = 512, 
        batch_per_step: int = 1, 
        model_lr: float = 1e-3, 
        reward_lr: float = 1e-3, 
        policy_lr: float = 3e-4, 
        value_lr: float = 3e-4, 
        max_grad_norm: float = 1.0, 
        batch_size: int = 256, 
        num_particles: int = 5, 
        horizon: int = 5, 
        **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self._epoch_size = self.hparams.batch_per_step * self.hparams.batch_size * self.hparams.steps_per_epoch
        self._memory = SLCOMemory(
            gamma=self.hparams.gamma, 
            env=gym.make(env, render_mode="human"),
            capacity=self.hparams.buffer_capacity, 
            epoch_size=self._epoch_size)
        # policy network
        self._policy = policy_map[policy_net](**self.env_info)
        self._policy_frozen = policy_map[value_net](**self.env_info)
        for param in self._policy_frozen.parameters():
            param.requires_grad = False
        self._hard_update(self._policy, self._policy_frozen)
        # value network
        self._value = value_map[value_net](**self.env_info)
        self._value_frozen = value_map[value_net](**self.env_info)
        for param in self._value_frozen.parameters():
            param.requires_grad = False
        self._hard_update(self._value, self._value_frozen)
        # model network
        self._model = model_map[model_net](**self.env_info)
        self._model_frozen = model_map[model_net](**self.env_info)
        for param in self._model_frozen.parameters():
            param.requires_grad = False
        self._hard_update(self._model, self._model_frozen)
        # reward network
        self._reward = reward_map[reward_net](**self.env_info)
        self._reward_frozen = reward_map[reward_net](**self.env_info)
        for param in self._reward_frozen.parameters():
            param.requires_grad = False
        self._hard_update(self._reward, self._reward_frozen)
        # memory reset and initial rollouts
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

    # http://proceedings.mlr.press/v70/asadi17a/asadi17a.pdf
    @staticmethod
    def mellowmax(input: th.Tensor, temp: float, dim: int, keepdim: bool = False):
        return temp * th.logsumexp(input/temp, dim=dim, keepdim=keepdim)
        # e, _ = input.max(dim=dim)
        return e

    def forward(self, observation: th.Tensor):
        distr = self._policy(observation)
        action = distr.rsample() #.clip(-1+1e-5, 1-1e-5)
        log_prob = distr.log_prob(action).sum(dim=-1)
        return action, log_prob, None

    @th.no_grad()
    def step(self, observation: np.ndarray):
        observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
        action_, log_prob_, _ = self.forward(observation_)
        action = action_.squeeze(0).cpu().numpy()
        log_prob = log_prob_.squeeze(0).cpu().numpy()
        return action, log_prob, None

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
        model_optim, reward_optim, policy_optim, value_optim = self.optimizers()
        # model learning
        model_optim.zero_grad()
        delta_observation_distr = self._model(observation, action)
        delta_observation_particles = delta_observation_distr.rsample((self.hparams.num_particles, ))
        model_entropy = - delta_observation_distr.log_prob(delta_observation_particles).mean(dim=0)
        info_delta_observation = - delta_observation_distr.log_prob(next_observation - observation)
        if self._model.independent_observations:
            info_delta_observation = info_delta_observation.sum(dim=-1)
            model_entropy = model_entropy.sum(dim=-1)
        model_loss = info_delta_observation.mean() - self.hparams.beta * model_entropy.mean()
        self.manual_backward(model_loss)
        model_optim.step()
        # reward learning
        reward_optim.zero_grad()
        reward_est = self._reward(observation, action)
        reward_target = reward 
        reward_loss = th.nn.functional.mse_loss(reward_est, reward_target) # next_value_error
        self.manual_backward(reward_loss)
        reward_optim.step()
        # model based artificial rollout
        policy_optim.zero_grad()
        observation_particles = th.stack(self.hparams.num_particles*[observation], dim=0) # num_particles, batch, nobs
        return_rollout = 0 # Implement
        action_distr = self._policy(observation) # num_particles, batch_size, nobs
        action_particles = action_distr.rsample((self.hparams.num_particles, )) # num_particles, batch_size, // nact
        policy_entropy = - action_distr.log_prob(action_particles).mean(dim=0) # batch_size, // nact
        if self._policy.independent_actions:
            policy_entropy = policy_entropy.sum(dim=-1) # num_particles, batch_size
        observation_particles = th.stack(self.hparams.num_particles*[observation], dim=0) # num_particles, batch, nobs
        reward_particles = self._reward_frozen(observation_particles, action_particles) # num_particles, batch_size
        d_observation_distr = self._model_frozen(observation_particles, action_particles)
        d_observation_particles = d_observation_distr.rsample()
        next_observation_particles = d_observation_particles + observation_particles
        observation_particles = next_observation_particles # for next time step
        last_value_particles = self._value_frozen(observation_particles)
        return_rollout = reward_particles + self.hparams.gamma * last_value_particles # num_particles, batch_size
        policy_loss = - return_rollout.mean() - self.hparams.alpha * policy_entropy.mean()
        self.manual_backward(policy_loss)
        policy_optim.step()
        # value learning
        value_optim.zero_grad()
        value_est = self._value(observation) # batch_size
        value_target = return_rollout.detach() # num_particles, batch_size
        value_loss = th.nn.functional.mse_loss(value_est.unsqueeze(0), value_target)
        self.manual_backward(value_loss)
        value_optim.step()
        # end of training step overall
        # self._soft_update(self._policy, self._policy_frozen)
        self._soft_update(self._value, self._value_frozen)
        self._soft_update(self._model, self._model_frozen)
        self._soft_update(self._reward, self._reward_frozen)
        # 
        info = {
            "policy_loss": policy_loss, 
            "value_loss": value_loss, 
            "reward_loss": reward_loss, 
            "model_loss": model_loss, 
            "policy_entropy": policy_entropy.mean(), 
            "model_entropy": model_entropy.mean(),
        }
        for key, value in info.items():
            self.log(key, value)
        return info

    def training_step_slco(self, batch, batch_index):
        observation, action, reward, next_observation, done, truncated = batch
        model_optim, reward_optim, policy_optim, value_optim = self.optimizers()
        # model learning
        model_optim.zero_grad()
        delta_observation_distr = self._model(observation, action)
        info_delta_observation = - delta_observation_distr.log_prob(next_observation - observation)
        if self._model.independent_observations:
            info_delta_observation = info_delta_observation.sum(dim=-1)
        model_loss = self.hparams.beta * info_delta_observation.mean()
        self.manual_backward(model_loss)
        # self.clip_gradients(model_optim, gradient_clip_val=self.hparams.max_grad_norm, gradient_clip_algorithm="norm")
        model_optim.step()
        # reward learning
        reward_optim.zero_grad()
        reward_est = self._reward(observation, action)
        reward_target = reward + self.hparams.beta * info_delta_observation.detach() # optimism on reward!
        reward_loss = th.nn.functional.mse_loss(reward_est, reward_target) # next_value_error
        self.manual_backward(reward_loss)
        reward_optim.step()
        # model based artificial rollout
        policy_optim.zero_grad()
        observation_particles = th.stack(self.hparams.num_particles*[observation], dim=0) # num_particles, batch, nobs
        return_rollout = 0 # Implement
        for i in range(self.hparams.horizon):
            action_distr = self._policy(observation_particles) # num_particles, batch_size, nobs
            action_particles = action_distr.rsample()
            entropy = - action_distr.log_prob(action_particles) # num_particles, batch_size, // nact
            if self._policy.independent_actions:
                entropy = entropy.sum(dim=-1) # num_particles, batch_size
            reward_particles = self._reward_frozen(observation_particles, action_particles) # num_particles, batch_size
            return_rollout += self.hparams.gamma**i * (reward_particles + self.hparams.alpha * entropy)
            d_observation_distr = self._model_frozen(observation_particles, action_particles)
            d_observation_particles = d_observation_distr.rsample()
            next_observation_particles = d_observation_particles + observation_particles
            observation_particles = next_observation_particles # for next time step
        # end of horizon
        last_value_particles = self._value_frozen(observation_particles)
        return_rollout += self.hparams.gamma**self.hparams.horizon * last_value_particles # num_particles, batch_size
        policy_loss = - return_rollout.mean()
        self.manual_backward(policy_loss)
        policy_optim.step()
        # value learning
        value_optim.zero_grad()
        value_est = self._value(observation) # batch_size    def training_step(self, batch, batch_index):
        observation, action, reward, next_observation, done, truncated = batch
        model_optim, reward_optim, policy_optim, value_optim = self.optimizers()
        # model learning
        model_optim.zero_grad()
        delta_observation_distr = self._model(observation, action)
        info_delta_observation = - delta_observation_distr.log_prob(next_observation - observation)
        if self._model.independent_observations:
            info_delta_observation = info_delta_observation.sum(dim=-1)
        model_loss = self.hparams.beta * info_delta_observation.mean()
        self.manual_backward(model_loss)
        # self.clip_gradients(model_optim, gradient_clip_val=self.hparams.max_grad_norm, gradient_clip_algorithm="norm")
        model_optim.step()
        # reward learning
        reward_optim.zero_grad()
        reward_est = self._reward(observation, action)
        reward_target = reward + self.hparams.beta * info_delta_observation.detach() # optimism on reward!
        reward_loss = th.nn.functional.mse_loss(reward_est, reward_target) # next_value_error
        self.manual_backward(reward_loss)
        reward_optim.step()
        # model based artificial rollout
        policy_optim.zero_grad()
        observation_particles = th.stack(self.hparams.num_particles*[observation], dim=0) # num_particles, batch, nobs
        return_rollout = 0 # Implement
        for i in range(self.hparams.horizon):
            action_distr = self._policy(observation_particles) # num_particles, batch_size, nobs
            action_particles = action_distr.rsample()
            entropy = - action_distr.log_prob(action_particles) # num_particles, batch_size, // nact
            if self._policy.independent_actions:
                entropy = entropy.sum(dim=-1) # num_particles, batch_size
            reward_particles = self._reward_frozen(observation_particles, action_particles) # num_particles, batch_size
            return_rollout += self.hparams.gamma**i * (reward_particles + self.hparams.alpha * entropy)
            d_observation_distr = self._model_frozen(observation_particles, action_particles)
            d_observation_particles = d_observation_distr.rsample()
            next_observation_particles = d_observation_particles + observation_particles
            observation_particles = next_observation_particles # for next time step
        # end of horizon
        last_value_particles = self._value_frozen(observation_particles)
        return_rollout += self.hparams.gamma**self.hparams.horizon * last_value_particles # num_particles, batch_size
        policy_loss = - return_rollout.mean()
        self.manual_backward(policy_loss)
        policy_optim.step()
        # value learning
        value_optim.zero_grad()
        value_est = self._value(observation) # batch_size
        value_target = return_rollout.detach() # num_particles, batch_size
        value_loss = th.nn.functional.mse_loss(value_est.unsqueeze(0), value_target)
        self.manual_backward(value_loss)
        value_optim.step()
        # end of training step overall
        self._soft_update(self._policy, self._policy_frozen)
        self._soft_update(self._value, self._value_frozen)
        self._soft_update(self._model, self._model_frozen)
        self._soft_update(self._reward, self._reward_frozen)
        # 
        # value_grad_norm = eval_grad_norm(self._value)
        # policy_grad_norm = eval_grad_norm(self._policy)
        # model_grad_norm = eval_grad_norm(self._model)
        # reward_grad_norm = eval_grad_norm(self._reward)
        info = {
            "policy_loss": policy_loss, 
            "value_loss": value_loss, 
            "reward_loss": reward_loss, 
            "model_loss": model_loss, 
            # "value_grad_norm": value_grad_norm, 
            # "policy_grad_norm": policy_grad_norm, 
            # "model_grad_norm": model_grad_norm, 
            # "reward_grad_norm": reward_grad_norm, 
        }
        for key, value in info.items():
            self.log(key, value)
        return info
        value_target = return_rollout.detach() # num_particles, batch_size
        value_loss = th.nn.functional.mse_loss(value_est.unsqueeze(0), value_target)
        self.manual_backward(value_loss)
        value_optim.step()
        # end of training step overall
        self._soft_update(self._policy, self._policy_frozen)
        self._soft_update(self._value, self._value_frozen)
        self._soft_update(self._model, self._model_frozen)
        self._soft_update(self._reward, self._reward_frozen)
        # 
        # value_grad_norm = eval_grad_norm(self._value)
        # policy_grad_norm = eval_grad_norm(self._policy)
        # model_grad_norm = eval_grad_norm(self._model)
        # reward_grad_norm = eval_grad_norm(self._reward)
        info = {
            "policy_loss": policy_loss, 
            "value_loss": value_loss, 
            "reward_loss": reward_loss, 
            "model_loss": model_loss, 
            # "value_grad_norm": value_grad_norm, 
            # "policy_grad_norm": policy_grad_norm, 
            # "model_grad_norm": model_grad_norm, 
            # "reward_grad_norm": reward_grad_norm, 
        }
        for key, value in info.items():
            self.log(key, value)
        return info

    def training_step_pg(self, batch, batch_index):
        observation, action, reward, next_observation, done, truncated = batch
        model_optim, reward_optim, policy_optim, value_optim = self.optimizers()
        # model learning
        model_optim.zero_grad()
        delta_observation_distr = self._model(observation, action)
        info_delta_observation = - delta_observation_distr.log_prob(next_observation - observation)
        if self._model.independent_observations:
            info_delta_observation = info_delta_observation.sum(dim=-1)
        model_loss = self.hparams.beta * info_delta_observation.mean()
        self.manual_backward(model_loss)
        #self.clip_gradients(model_optim, gradient_clip_val=self.hparams.max_grad_norm, gradient_clip_algorithm="norm")
        model_optim.step()
        # reward learning
        reward_optim.zero_grad()
        reward_est = self._reward(observation, action)
        reward_target = reward + self.hparams.beta * info_delta_observation.detach() # pessimism on reward!
        reward_loss = th.nn.functional.mse_loss(reward_est, reward_target) # next_value_error
        self.manual_backward(reward_loss)
        #self.clip_gradients(reward_optim, gradient_clip_val=self.hparams.max_grad_norm, gradient_clip_algorithm="norm")
        reward_optim.step()
        # model based artificial rollout
        value_optim.zero_grad()
        policy_optim.zero_grad()
        action_distr = self._policy(observation) # batch_size, nact
        action_samples = action_distr.rsample((self.hparams.num_particles, )) # num_particles, batch_size, nact
        action_particles = action_samples.detach() # detached actions / w.o gradient
        entropy = - action_distr.log_prob(action_samples) 
        info_action_particles = - action_distr.log_prob(action_particles)
        if self._policy.independent_actions:
            entropy = entropy.sum(dim=-1)
            info_action_particles = info_action_particles.sum(dim=-1)
        with th.no_grad():
            observation_particles = th.stack(self.hparams.num_particles*[observation], dim=0) # num_particles, batch, nobs
            reward_particles = self._reward_frozen(observation_particles, action_particles) # num_particles, batch_size
            d_observation_distr = self._model_frozen(observation_particles, action_particles) # num_particles, batch_size
            d_observation_particles = d_observation_distr.rsample() # num_particles, batch_size
            next_observation_particles = d_observation_particles + observation_particles # num_particles, batch_size
            # end of horizon
            next_value_particles = self._value_frozen(next_observation_particles) # num_particles, batch_size
            value_target = reward_particles + self.hparams.alpha * info_action_particles.detach() + self.hparams.gamma * next_value_particles
        value_est = self._value(observation) # batch_size
        value_loss = th.nn.functional.mse_loss(value_est.unsqueeze(0), value_target)
        advantage_particles = reward_particles + self.hparams.gamma * next_value_particles - value_est.unsqueeze(0).detach()
        policy_obj = info_action_particles * advantage_particles.detach() - self.hparams.alpha * entropy
        policy_loss = self.hparams.alpha * policy_obj.mean()
        self.manual_backward(value_loss)
        #self.clip_gradients(value_optim, gradient_clip_val=self.hparams.max_grad_norm, gradient_clip_algorithm="norm")
        self.manual_backward(policy_loss)
        #self.clip_gradients(policy_optim, gradient_clip_val=self.hparams.max_grad_norm, gradient_clip_algorithm="norm")
        value_optim.step()
        policy_optim.step()
        # end of training step overall
        self._soft_update(self._policy, self._policy_frozen)
        self._soft_update(self._value, self._value_frozen)
        self._soft_update(self._model, self._model_frozen)
        self._soft_update(self._reward, self._reward_frozen)
        # 
        value_grad_norm = eval_grad_norm(self._value)
        policy_grad_norm = eval_grad_norm(self._policy)
        model_grad_norm = eval_grad_norm(self._model)
        reward_grad_norm = eval_grad_norm(self._reward)
        ##### print(value_grad_norm)
        ##### print(policy_grad_norm)
        ##### print(model_grad_norm)
        ##### print(reward_grad_norm)
        ##### print(entropy.mean())
        ##### print(info_action_particles.detach().mean())
        ##### print(observation_particles.std())
        ##### print(reward_particles.std())
        ##### print(next_observation_particles.std())
        ##### print(next_value_particles.mean())
        ##### print(next_value_particles.std())
        ##### print(value_target.mean())
        ##### print(value_target.std())
        ##### print(value_est.mean())
        ##### print(value_est.std())
        ##### print(value_loss)
        ##### print(advantage_particles.mean())
        ##### print(advantage_particles.std())
        ##### print(policy_obj.mean())
        ##### print(policy_obj.std())
        ##### print(policy_loss)
        ##### print("lgfdskjksdjfhgkjdsfhgkjfds")
        ##### if info_action_particles.detach().mean().isnan():
        #####     th.save(info_action_particles, "info_action_particles")
        #####     th.save(entropy, "entropy")
        #####     th.save(action_samples, "action_samples")
        #####     th.save(action_particles, "action_particles")
        #####     th.save(action_distr, "action_distr")
        #####     # info_action_particles = th.load("info_action_particles")
        #####     # entropy = th.load("entropy")
        #####     # action_samples = th.load("action_samples")
        #####     # action_particles = th.load("action_particles")
        #####     # action_distr = th.load("action_distr")
        #####     print("fucking saved!")
        info = {
            "policy_loss": policy_loss, 
            "value_loss": value_loss, 
            "reward_loss": reward_loss, 
            "model_loss": model_loss, 
            "policy_entropy": entropy.mean(), 
            "value_grad_norm": value_grad_norm, 
            "policy_grad_norm": policy_grad_norm, 
            "model_grad_norm": model_grad_norm, 
            "reward_grad_norm": reward_grad_norm, 
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
        """Initialize AdamW optimizer."""
        model_optim = AdamW(self._model.parameters(), lr=self.hparams.model_lr, weight_decay=1e-4)
        reward_optim = AdamW(self._reward.parameters(), lr=self.hparams.reward_lr, weight_decay=1e-4)
        policy_optim = AdamW(self._policy.parameters(), lr=self.hparams.policy_lr, weight_decay=1e-4)
        value_optim = AdamW(self._value.parameters(), lr=self.hparams.value_lr, weight_decay=1e-4)
        return model_optim, reward_optim, policy_optim, value_optim

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
        parser.add_argument("--policy_net", type=str, default="continuous_mlp2")
        parser.add_argument("--value_net", type=str, default="continuous_mlp2")
        parser.add_argument("--model_net", type=str, default="continuous_mlp2")
        parser.add_argument("--reward_net", type=str, default="continuous_mlp2")
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--beta", type=float, default=0.05)
        parser.add_argument("--buffer_capacity", type=int, default=1000000)
        parser.add_argument("--tau", type=float, default=5e-3)
        parser.add_argument("--start_steps", type=int, default=10000)
        parser.add_argument("--steps_per_epoch", type=int, default=512)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--model_lr", type=float, default=1e-3)
        parser.add_argument("--reward_lr", type=float, default=1e-3)
        parser.add_argument("--policy_lr", type=float, default=3e-4)
        parser.add_argument("--value_lr", type=float, default=3e-4)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_particles", type=int, default=5)
        parser.add_argument("--horizon", type=int, default=5)
        return parser

if __name__=="__main__":
    pass
