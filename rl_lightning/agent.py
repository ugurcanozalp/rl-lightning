
import os
from collections import OrderedDict, deque, namedtuple
import random
from typing import Iterator, List, Tuple, Callable, Any, Dict, Union

import numpy as np
import torch as th


class Policy(object):

    def forward(self, observation: th.Tensor):
        raise NotImplementedError
        # return action, log_prob, value

    @th.no_grad()
    def step(self, observation: np.ndarray):
        raise NotImplementedError
        # return action, log_prob, value

    def reset(self):
        raise NotImplementedError


class Model(object):

    def forward(self, action: th.Tensor):
        raise NotImplementedError
        # return next_state, reward, done

    @th.no_grad()
    def step(self, action: np.ndarray):
        raise NotImplementedError
        # return next_state, reward, done

    def reset(self):
        raise NotImplementedError 
        # self.state = []


class CommonValuePolicy(Policy):
    
    def __init__(self, piv_cls: th.nn.Module):
        self.piv = piv_cls()

    def forward(self, observation):
        distr, value = self.piv(observation)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        return action, log_prob, value        

    def reset(self):
        pass


class ValuePolicy(Policy):

    def __init__(self, pi_cls: th.nn.Module, v_cls: th.nn.Module):
        self.pi = pi_cls()
        self.v = v_cls()

    def forward(self, observation):
        distr = self.pi(observation)
        value = self.v(observation)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        return action, log_prob, value

    @th.no_grad()
    def step(self, observation):
        observation_ = th.from_numpy(observation).unsqueeze(0)
        action_, log_prob_, value_ = self.forward(observation_)
        action = action_.squeeze(0).cpu().numpy()
        log_prob = log_prob_.squeeze(0).cpu().numpy()
        value = value_.squeeze(0).cpu().numpy()
        return action, log_prob, value

    def reset(self):
        pass


class QValuePolicy(Policy):
    
    def __init__(self, pi_cls: th.nn.Module, q_cls: th.nn.Module):
        self.pi = pi_cls()
        self.q = q_cls()

    def forward(self, observation):
        distr = self.pi(observation)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        value = self.q(observation, action)
        return action, log_prob, value

    @th.no_grad()
    def step(self, observation):
        observation_ = th.from_numpy(observation).unsqueeze(0)
        action_, log_prob_, value_ = self.forward(observation_)
        action = action_.squeeze(0).cpu().numpy()
        log_prob = log_prob_.squeeze(0).cpu().numpy()
        value = value_.squeeze(0).cpu().numpy()
        return action, log_prob, value

    def reset(self):
        pass


class VanillaModel(object):

    def __init__(self, model_cls: th.nn.Module):
        self.model = model_cls()

    def forward(self, action: th.Tensor):
        distr, reward, done = self.model(self.state, action)
        next_state = distr.rsample()
        log_prob = distr.log_prob(next_state)
        return next_state, log_prob, reward, done

    @th.no_grad()
    def step(self, action: np.ndarray):
        raise NotImplementedError
        # return next_state, reward, done

    def reset(self):
        raise NotImplementedError 
        # self.state = []