
import torch as th
from torch import nn
from .heads import *


class ContinuousMLPModelReward(nn.Module):
    independent_observations = True
    def __init__(self, observation_shape, action_shape, **kwargs):
        super(ContinuousMLPModelReward, self).__init__()
        num_inputs = observation_shape[0] + action_shape[0]
        self.fc1 = nn.Sequential(nn.Linear(num_inputs, 512), nn.LayerNorm(512), nn.GELU())
        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU())
        self.ds_fc = nn.Linear(512, 2*observation_shape[0])
        self.ds_head = GaussianHead(observation_shape[0])
        self.r_fc = nn.Linear(512, 1)

    def forward(self, observation, action):
        x = th.concat([observation, action], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        ds = self.ds_fc(x)
        ds_distr = self.ds_head(ds)
        r = self.r_fc(x)
        return ds_distr, r.squeeze(-1)
