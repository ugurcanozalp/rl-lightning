
import math

import torch as th
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus


LOG2 = math.log(2.)

class StableTanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.

    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.

    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _call(self, x):
        finfo = th.finfo(x.dtype)
        return x.tanh().clamp(-1+finfo.eps, 1-finfo.eps)

    def _inverse(self, y):
        # We clamp to the boundary here.
        finfo = th.finfo(y.dtype)
        return th.atanh(y.clamp(-1+finfo.eps, 1-finfo.eps))

    def log_abs_det_jacobian(self, x, y):
        # finfo = th.finfo(y.dtype)
        return (1 - y*y).log() # + 2*finfo.eps


