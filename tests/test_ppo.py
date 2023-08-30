
import pytest

import numpy as np

from rl_lightning.ppo import PPOMemory, PPO

mem = PPOMemory(gamma=0.99, lambd=0.97, capacity=100, epoch_size=40)

for i in range(50):
    observation = np.random.randn(10)
    action = np.random.rand(3)
    reward = 1
    next_observation = np.random.randn(10)
    done = np.random.rand(1)>0.5
    truncated = np.random.rand(1)>0.5
    log_prob = np.random.randn(1)
    value = np.random.randn(1)
    mem.step(
        observation,
        action,
        reward,
        next_observation,
        done,
        truncated,
        log_prob,
        value)

mem.finish_episode()