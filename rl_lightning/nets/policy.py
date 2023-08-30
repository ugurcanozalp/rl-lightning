
import torch as th
from torch import nn
from .heads import *


class ContinuousMLPPolicy(nn.Module):
    independent_actions = True
    def __init__(self, observation_shape, action_shape, **kwargs):
        super(ContinuousMLPPolicy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(observation_shape[0], 256), nn.LayerNorm(256), nn.GELU())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU())
        self.p_fc = nn.Linear(256, 2*action_shape[0])
        self.p_head = SquashedGaussianHead(action_shape[0])
        # self.p_head = GaussianHead(action_shape[0])

    def forward(self, observation):
        x = observation
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.p_fc(x)
        x = self.p_head(x)
        return x
