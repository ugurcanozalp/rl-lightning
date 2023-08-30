
from .policy import ContinuousMLPPolicy
from .value import ContinuousMLPValue
from .qvalue import ContinuousMLPQValue
from .policy_value import ContinuousMLPPolicyValue
from .model import ContinuousMLPModel
from .reward import ContinuousMLPReward
from .model_reward import ContinuousMLPModelReward

policy_map = {
    "continuous_mlp2": ContinuousMLPPolicy
}

value_map = {
    "continuous_mlp2": ContinuousMLPValue
}

qvalue_map = {
    "continuous_mlp2": ContinuousMLPQValue
}

policy_value_map = {
    "continuous_mlp2": ContinuousMLPPolicyValue
}

model_map = {
    "continuous_mlp2": ContinuousMLPModel
}

reward_map = {
    "continuous_mlp2": ContinuousMLPReward
}

model_reward_map = {
    "continuous_mlp2": ContinuousMLPModelReward
}