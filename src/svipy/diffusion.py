import abc
import math
from typing import Tuple  # Optional, Iterable, Union, overload, Tuple, List
# import numpy as np

import torch
from torchdiffeq import odeint_adjoint

from svipy.model import baseTorchModel, baseLossTracker
from svipy.normflow import timeConditionedField


#################################
#         Flow Matching         #
#################################
class conditionalPath(torch.nn.Module, abc.ABC):
    def __init__(self):
        super(conditionalPath, self).__init__()

    @abc.abstractmethod
    def forward(self, x0, x1, t) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x_t, u_t): interpolated point and its target velocity."""
        pass

    def generate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        return self.forward(x0, x1, t)[0]

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:`
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        return self.forward(x0, x1, t)[1]

    @staticmethod
    def _expand(t, x):
        assert t.dim() == 1 and t.shape[0] == x.shape[0]
        return t.view(t.shape[0], *([1] * (x.dim() - 1))) if t.dim() == 1 else t


class linearConditionalPath(conditionalPath):
    def __init__(self, minSigma=1e-4):
        super(linearConditionalPath, self).__init__()
        self.minSigma = minSigma

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        tx = self._expand(t, x0)
        return tx * x1 + (1 - (1 - self.minSigma) * tx) * x0, x1 - (1 - self.minSigma) * x0


class conditionalFlowMatcher(baseTorchModel):
    def __init__(self, pathGenerator: conditionalPath, velocityField: timeConditionedField):
        super(conditionalFlowMatcher, self).__init__()
        self.pathGenerator = pathGenerator
        self.velocityField = velocityField

    def computeLoss(self, data) -> dict:
        X1 = data.to(self.device)
        X0 = torch.randn_like(X1, device=self.device)
        t = torch.rand(X1.shape[0], device=self.device)

        Xt = self.pathGenerator.generate(X0, X1, t)
        Ut = self.pathGenerator.velocity(X0, X1, t)

        Vt = self.velocityField(Xt, t)

        totalLoss = torch.mean(torch.sum(torch.square(Vt - Ut).flatten(start_dim=1), dim=1))

        return {"totalLoss": totalLoss}


#################################
#           Diffusion           #
#################################

class varPreservingConditionalPath(conditionalPath):
    def __init__(self):
        super(varPreservingConditionalPath, self).__init__()

    @abc.abstractmethod
    def alpha(self, t):
        pass

    @abc.abstractmethod
    def dalpha(self, t):
        pass

    def sigma(self, t):
        return torch.sqrt(1. - self.alpha(t) * self.alpha(t))

    def dsigma(self, t):
        return -self.alpha(t) * self.dalpha(t) / self.sigma(t)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        tx = self._expand(t, x0)
        return self.alpha(tx) * x1 + self.sigma(tx) * x0, self.dalpha(tx) * x1 + self.dsigma(tx) * x0


class varPreservingConditionalPathTrigonometric(varPreservingConditionalPath):
    def __init__(self):
        super(varPreservingConditionalPathTrigonometric, self).__init__()

    def alpha(self, t):
        return torch.cos(t * torch.pi / 2.)

    def dalpha(self, t):
        return -torch.sin(t * torch.pi / 2.) * torch.pi / 2.

    def sigma(self, t):
        return torch.sin(t * torch.pi / 2.)

    def dsigma(self, t):
        return torch.cos(t * torch.pi / 2.) * torch.pi / 2.


class varPreservingConditionalPathDDPMCosine(varPreservingConditionalPath):
    def __init__(self, s=0.008):
        super().__init__()
        self.s = s
        self.a = 1. / (1 + self.s) * torch.pi / 2
        self.b = self.s * self.a
        self.cosB = math.cos(self.b)

    def _g(self, t):
        # return (t + self.s) / (1 + self.s) * torch.pi / 2
        return self.a * t + self.b

    def alpha(self, t):
        return torch.cos(self._g(t)) / self.cosB

    def dalpha(self, t):
        return -torch.sin(self._g(t)) * self.a / self.cosB


class varPreservingConditionalPathLinear(varPreservingConditionalPath):
    def __init__(self, betaMin=0.1, betaMax=20.0):
        super(varPreservingConditionalPathLinear, self).__init__()
        self.betaMin = betaMin
        self.betaMax = betaMax

    def beta(self, t):
        return self.betaMin + (self.betaMax - self.betaMin) * t

    def alpha(self, t):
        return torch.exp(-self.betaMin * t / 2.0 - (self.betaMax - self.betaMin) * t * t / 4.0)

    def dalpha(self, t):
        return - self.beta(t) * self.alpha(t) / 2.0

