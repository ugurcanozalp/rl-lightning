
import torch as th
from torch import nn
from torch.distributions import Distribution, Normal, Categorical, TransformedDistribution, MultivariateNormal 
from torch.distributions.transforms import TanhTransform, SigmoidTransform, AffineTransform, ComposeTransform
from .utils import StableTanhTransform


class GaussianHead(nn.Module):
	def __init__(self, n):
		super(GaussianHead, self).__init__()
		self._n = n

	def forward(self, x):
		mean = x[...,:self._n]
		logvar = x[...,self._n:]
		std = logvar.clip(-10, 2).exp() 
		dist = Normal(mean, std, validate_args=True)
		return dist


class SquashedGaussianHead(nn.Module):
	def __init__(self, n):
		super(SquashedGaussianHead, self).__init__()
		self._n = n

	def forward(self, x):
		# bt means before tanh
		mean_bt = x[...,:self._n] 
		logvar_bt = x[...,self._n:] 
		std_bt = logvar_bt.clip(-10, 2).exp() 
		dist_bt = Normal(mean_bt, std_bt, validate_args=True)
		transform = TanhTransform(cache_size=1)
		dist = TransformedDistribution(dist_bt, [transform], validate_args=True)
		return dist


class SquashedGaussianHeadVaryingBounds(nn.Module):
	def __init__(self, n, lower_bound, upper_bound):
		super(SquashedGaussianHeadVaryingBounds, self).__init__()
		self._n = n
		self._lower_bound = th.tensor(lower_bound, dtype=th.float)
		self._upper_bound = th.tensor(upper_bound, dtype=th.float)
		self._span = self._upper_bound - self._lower_bound

	def forward(self, x):
		# bt means before tanh
		mean_bt = x[...,:self._n] 
		logvar_bt = x[...,self._n:] 
		std_bt = logvar_bt.clip(-10, 2).exp() 
		dist_bt = Normal(mean_bt, std_bt, validate_args=True)
		transform = ComposeTransform(
			[
				AffineTransform(0., self._span), 
				SigmoidTransform(), 
				AffineTransform(self._lower_bound, self._span)
			]
		)
		dist = TransformedDistribution(dist_bt, [transform], validate_args=True)
		return dist


class CategoricalHead(nn.Module):
	def __init__(self, n):
		super(CategoricalHead, self).__init__()
		self._n = n

	def forward(self, x):
		logit = x
		probs = nn.functional.softmax(logit)
		dist = Categorical(probs, validate_args=True)
		return dist


class DeterministicHead(nn.Module):
	def __init__(self, n):
		super(DeterministicHead, self).__init__()
		self._n = n

	def forward(self, x):
		mean = x
		y = mean
		dist = None
		return y

