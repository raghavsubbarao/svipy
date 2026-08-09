import abc
import math
# from typing import Optional, Iterable, Union, overload, Tuple, List
# import numpy as np

import torch
from torchdiffeq import odeint_adjoint

from svipy.model import baseTorchModel, baseLossTracker


#################################
#         Flow Matching         #
#################################
class conditionalPath(torch.nn.Module, abc.ABC):
    def __init__(self):
        super(conditionalPath, self).__init__()

    @abc.abstractmethod
    def sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        pass

    @abc.abstractmethod
    def target(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:`
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        pass


class linearConditionalPath(conditionalPath):
    def __init__(self):
        super(linearConditionalPath, self).__init__()

    def sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        return (1 - t) * x0 + t * x1

    def target(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:`
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        return x1 - x0


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

    def sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        return self.alpha(t) * x1 + self.sigma(t) * x0

    def target(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: B x ... tensor where B=batch_size
        :param x1:
        :param t:`
        :return: tensor of size (B,) of inverse log determinant of the Jacobians
        """
        return self.dalpha(t) * x1 + self.dsigma(t) * x0


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
