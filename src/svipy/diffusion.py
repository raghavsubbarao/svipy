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


class reversedConditionalPath(conditionalPath):
    """
    Presents a wrapped conditionalPath under the opposite time convention -
    t=0 becomes t=1 and vice versa - without touching the wrapped path's own
    formulas. Used to adapt paths written in the standard diffusion
    convention (t=0 data, t=1 noise - e.g. varPreservingConditionalPath and
    its subclasses) to the flow matching convention and
    continuousNormFlow.generate()/interpolate() assume (t=0 noise, t=1
    data), while leaving the wrapped path available in its native form for
    diffusion-specific use (noise schedules, SNR weighting, etc.) that wants
    the original convention.
    """
    def __init__(self, path: conditionalPath):
        super(reversedConditionalPath, self).__init__()
        self.path = path

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # t is unexpanded (B,) here - the wrapped path does its own _expand
        xt, ut = self.path(x0, x1, 1. - t)
        return xt, -ut


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

        Xt, Ut = self.pathGenerator(X0, X1, t)
        Vt = self.velocityField(Xt, t)

        totalLoss = torch.mean(torch.sum(torch.square(Vt - Ut).flatten(start_dim=1), dim=1))

        return {"totalLoss": totalLoss}


#################################
#           Diffusion           #
#################################

class varPreservingConditionalPath(conditionalPath):
    def __init__(self, target='score'):
        super(varPreservingConditionalPath, self).__init__()

        assert target in ['velocity', 'score']
        self.target = target

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

    def forward(self, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x1: noise (for diffusion/score matching) - B x ... where B=batch_size
        :param x0: data (for diffusion/score matching) - B x ... where B=batch_size
        :param t:t=0 is clean data, t=1 is noise - the standard diffusion
                 convention, matching alpha/dalpha/sigma/dsigma and the papers
                 they're taken from. This is the OPPOSITE of linearConditionalPath
                 (and of conditionalFlowMatcher/continuousNormFlow, which assume
                 t=0 is noise, t=1 is data) - wrap with reversedConditionalPath to
                 present this path for flow matching
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        tx = self._expand(t, x1)
        xt = self.alpha(tx) * x0 + self.sigma(tx) * x1
        if self.target == 'velocity':
            target = self.dalpha(tx) * x0 + self.dsigma(tx) * x1
        elif self.target == 'score':
            # the score is given by -x1/sigma(t), but we return -x0 as
            # the regression target. this is to avoid the loss driven by
            # regions where sigma(t)->0. therefore we match the noise
            # directly and then return the score only at inference time
            target = - x1
        else:
            raise Exception(f'Unknown target type: {self.target}')

        return xt, target


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


class conditionalScoreMatcher(baseTorchModel):
    def __init__(self, pathGenerator: conditionalPath, scoreField: timeConditionedField):
        super(conditionalFlowMatcher, self).__init__()
        self.pathGenerator = pathGenerator
        self.scoreField = scoreField

    def computeLoss(self, data) -> dict:
        X0 = data.to(self.device)
        X1 = torch.randn_like(X1, device=self.device)
        t = torch.rand(X0.shape[0], device=self.device)

        Xt, Ut = self.pathGenerator(X1, X0, t)
        Vt = self.scoreField(Xt, t)

        totalLoss = torch.mean(torch.sum(torch.square(Vt - Ut).flatten(start_dim=1), dim=1))

        return {"totalLoss": totalLoss}


if __name__ == "__main__":
    pass